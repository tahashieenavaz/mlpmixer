import torch
from typing import Type
from .MLPBlock import MLPBlock


class MixerBlock(torch.nn.Module):
    def __init__(
        self,
        *,
        sequence_length: int,
        channels: int,
        tokens_mlp_dimension: int,
        channels_mlp_dimension: int,
        channel_mixing_activation: Type[torch.nn.Module],
        token_mixing_activation: Type[torch.nn.Module],
    ):
        super().__init__()
        self.abel = torch.nn.LayerNorm(channels)
        self.cain = torch.nn.LayerNorm(channels)
        self.token_mixing = MLPBlock(
            input_dimension=sequence_length,
            hidden_dimension=tokens_mlp_dimension,
            activation=channel_mixing_activation,
        )
        self.channel_mixing = MLPBlock(
            input_dimension=channels,
            hidden_dimension=channels_mlp_dimension,
            activation=token_mixing_activation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        # token mixing
        x = self.abel(x)
        x = x.transpose(1, 2)
        x = self.token_mixing(x)
        x = x.transpose(1, 2)
        x = residual + x

        # channel mixing
        residual = x
        x = self.cain(x)
        x = self.channel_mixing(x)
        return residual + x
