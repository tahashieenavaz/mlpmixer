import torch


class HeadModule(torch.nn.Module):
    def __init__(self, *, hidden_dimension: int, num_classes: int):
        super().__init__()
        self.linear = torch.nn.Linear(hidden_dimension, num_classes)
        torch.nn.init.zeros_(self.linear.weight)
        torch.nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)
