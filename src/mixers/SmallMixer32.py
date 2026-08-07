import torch
from .modules import MLPMixer


class SmallMixer32(MLPMixer):
    def __init__(
        self,
        *,
        num_classes: int,
        image_size: int,
    ):
        super().__init__(
            image_size=image_size,
            num_classes=num_classes,
            num_blocks=8,
            hidden_dimension=512,
            channels_mlp_dimension=2048,
            tokens_mlp_dimension=256,
            patch_size=32,
            image_channels=3,
        )


if __name__ == "__main__":
    model = SmallMixer32(num_classes=10, image_size=224)
    images = torch.randn(1, 3, 224, 224)
    predictions = model(images)

    assert predictions.shape[-1] == 10

    parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters (in millions): {parameters / pow(10, 6)}")
    print(f"Prediction shape: {predictions.shape}")
