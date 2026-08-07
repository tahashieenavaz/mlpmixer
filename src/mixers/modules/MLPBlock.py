import torch
from typing import Type


class MLPBlock(torch.nn.Module):
    def __init__(
        self,
        *,
        input_dimension: int,
        hidden_dimension: int,
        activation: Type[torch.nn.Module],
    ):
        super().__init__()
        self.mu = torch.nn.Linear(input_dimension, hidden_dimension)
        self.nu = torch.nn.Linear(hidden_dimension, input_dimension)
        self.xi = activation()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mu(x)
        x = self.xi(x)
        x = self.nu(x)
        return x
