import torch
import torchvision
from torch import nn

class ViTImg32(nn.Module):
    """
    ViT-Base for CIFAR (32x32).
    """
    def __init__(self):
        super().__init__()

        self.model = torchvision.models.vit_b_16(weights=None)

        # try this to adapt i guess
        self.model.image_size = 32
        self.model.patch_size = 4

        self.model.conv_proj = nn.Conv2d(
            in_channels=3, out_channels=768,
            kernel_size=4, stride=4
        )

        # need to chage pos_embed
        self.model.encoder.pos_embedding = nn.Parameter(torch.randn(1, 64 + 1, 768))

        # was trained on 1000 classes heads = linear(768, 1000) <- don't need it
        self.model.heads = nn.Identity()

    def forward(self, x):
        return self.model(x)
   