"""Demo Mechanic learning rate scale tuner."""

# flake8: noqa=DCO010
# pylint: disable=invalid-name

from dataclasses import dataclass
from typing import Optional

import matplotlib.pyplot as plt
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from src.examples.common import set_seed
from src.examples.fully_connected.fully_connected import FullyConnected
from src.manic import Mechanic, Updater


@dataclass
class DemoConfig:
    """
    Dataclass for demo configuration.

    Args:
        num_samples: Number of samples in dataset.
        input_dim: Input dimension.
        noise_std: Output noise standard deviation.
        batch_size: Batch size in training.
        num_epochs: Number of training epochs.
    """

    num_samples: int = 1000
    input_dim: int = 10
    noise_std: float = 0.1
    batch_size: int = 32
    num_epochs: int = 200


class SyntheticRegressionDataset(Dataset):
    """
    Synthetic regression dataset.

    Args:
        config: Demo configuration.
    """

    def __init__(self, config: DemoConfig) -> None:
        # Weight and bias for affine function
        self.A = torch.randn(1, config.input_dim)
        self.b = torch.randn(1)

        # Input data
        self.x = torch.randn(config.num_samples, config.input_dim)

        # Output data is affine transformation of input data, plus noise
        self.y = nn.functional.linear(self.x, self.A, self.b)  # pylint: disable=not-callable
        self.y = self.y + (config.noise_std * torch.randn_like(self.y))

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        return self.x[idx], self.y[idx]


class Trainer:
    """
    Training loop.

    Args:
        device: Device to use for training (default = None).
        config: Demo configuration (default = None).
    """

    def __init__(self, device: Optional[str] = None, config: Optional[DemoConfig] = None) -> None:
        # Helper function to get current device name
        def get_device() -> torch.device:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Get default configuration, if necessary
        self.device = get_device() if device is None else device
        self.config = DemoConfig() if config is None else config

        # Make dataset and dataloader
        self.dataset = SyntheticRegressionDataset(self.config)
        self.dataloader = DataLoader(self.dataset, batch_size=self.config.batch_size, shuffle=True)

        # Make model and loss criterion
        # - Note: Model does not apply ReLU activation at final layer
        self.model = FullyConnected(self.config.input_dim, [64, 32], 1, 0.0, False).to(device)
        self.criterion = nn.MSELoss()

        # Make Updater with SGD and static LR scheduler
        self.updater = Updater(torch.optim.SGD(self.model.parameters()))

        # Make Mechanic
        self.mechanic = Mechanic(self.updater)

        # Metrics to track in each epoch
        self.train_losses: list[float] = []
        self.lr_scales: list[float] = []

    def train(self) -> tuple[list[float], list[float]]:
        """
        Run training loop.
        """
        print(f"Training on {self.device}")

        self.model.train()

        for epoch in range(self.config.num_epochs):
            epoch_loss = 0.0
            epoch_lr_scale = 0.0

            for x, y in self.dataloader:
                # Move data to device
                x = x.to(self.device)
                y = y.to(self.device)

                # Zero model parameter gradients
                self.updater.base_optimizer.zero_grad()

                # Run forward pass
                y_hat = self.model(x)

                # Compute loss
                loss = self.criterion(y_hat, y)

                # Run backward pass
                loss.backward()

                # Run one Mechanic step
                self.mechanic.step()

                # Run one updater step
                self.updater.step()

                # Accumulate loss and LR scale for this epoch
                epoch_loss += loss.item()
                epoch_lr_scale += self.mechanic.get_last_lr()[0]

            # Record metrics for this epoch
            epoch_loss = epoch_loss / len(self.dataloader)
            epoch_lr_scale = epoch_lr_scale / len(self.dataloader)

            self.train_losses.append(epoch_loss)
            self.lr_scales.append(epoch_lr_scale)

            # Console reporting
            print(
                f"epoch {epoch + 1}: " f"loss: {epoch_loss:.4f}, " f"LR scale: {epoch_lr_scale:.6f}"
            )

        return self.train_losses, self.lr_scales

    def plot_metrics(self, train_losses: list[float], lr_scales: list[float]) -> None:
        """
        Visualize metrics.

        Args:
            train_losses: Average losses per epoch.
            lr_scales: Average learning rate scale per epoch.
        """
        _, (ax1, ax2) = plt.subplots(1, 2, sharex=True)

        # Plot average losses for each epoch
        ax1.plot(train_losses, label="loss", color="blue")
        ax1.set_xlabel("epoch")
        ax1.set_ylabel("loss")
        ax1.set_title("Average loss per epoch")

        # Plot learning rates after each epoch
        ax2.plot(lr_scales, label="LR scale", color="red")
        ax2.set_xlabel("epoch")
        ax2.set_ylabel("LR scale")
        ax2.set_title("Average LR scale per epoch")

        plt.tight_layout()
        plt.show()


def run_demo() -> None:
    """Run demo for a fully-connected neural network."""
    trainer = Trainer()
    train_losses, learning_rates = trainer.train()
    trainer.plot_metrics(train_losses, learning_rates)


if __name__ == "__main__":
    set_seed(11)
    torch.set_default_dtype(torch.float64)
    plt.rcParams["text.usetex"] = True
    run_demo()
