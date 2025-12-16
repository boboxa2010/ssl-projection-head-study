import torch
from torch import nn
from src.model.encoder import ResNetImg32, ViTImg32
from src.model.heads import UniversalProjectionHead


class BarlowTwins(nn.Module):
    """
    Barlow Twins realization with ability to choose encoders and heads
    """

    def __init__(
            self,
            encoder: str = 'ResNet32',  # or ViT32
            mode: str = 'mlp_barlow',  # mlp_barlow or fixed
            kappa: float | None = None  # only in fixed head
    ):
        super().__init__()
        if encoder == 'ResNet32':
            self.encoder = ResNetImg32()
            input_dim = 512
        elif encoder == 'ViT32':
            self.encoder = ViTImg32()
            input_dim = 768
            # output - 768 maybe change ViT inside more?
        else:
            raise ValueError(f"Unknown encoder: {encoder}")

        self.projection_head = UniversalProjectionHead(input_dim, input_dim, 512, mode, kappa)

    def forward(self, x: torch.Tensor) -> dict:
        features = self.encoder(x)  # h
        projections = self.projection_head(features)  # z

        return {
            "features": features,
            "projections": projections
        }

    def __str__(self):
        """
        Model prints with the number of parameters.
        """
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info