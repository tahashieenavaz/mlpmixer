import torch
from typing import Type
from .modules import MLPMixer


class BaseMixer16(MLPMixer):
    def __init__(
        self,
        *,
        num_classes: int,
        image_size: int,
        channel_mixing_activation: Type[torch.nn.Module] = torch.nn.GELU,
        token_mixing_activation: Type[torch.nn.Module] = torch.nn.GELU,
    ):
        super().__init__(
            image_size=image_size,
            num_classes=num_classes,
            num_blocks=12,
            hidden_dimension=768,
            channels_mlp_dimension=3072,
            tokens_mlp_dimension=384,
            patch_size=16,
            image_channels=3,
            channel_mixing_activation=channel_mixing_activation,
            token_mixing_activation=token_mixing_activation,
        )


if __name__ == "__main__":
    model = BaseMixer16(num_classes=10, image_size=224)
    images = torch.randn(1, 3, 224, 224)
    predictions = model(images)

    assert predictions.shape[-1] == 10

    parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters (in millions): {parameters / pow(10, 6)}")
    print(f"Prediction shape: {predictions.shape}")
