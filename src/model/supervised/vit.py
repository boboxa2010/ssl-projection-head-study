import torchvision
from torch import nn


class ViTModel(nn.Module):
    """
    Wrapper over Vision Transformer (ViT) from torchvision
    """

    def __init__(
        self,
        n_class: int,
        in_channels: int = 3,
        image_size: int = 32,
        patch_size: int = 4,
    ):
        """
        Args:
            n_class (int): number of classes.
            in_channels (int): number of input channels.
            image_size (int): input image size (must match ViT config).
            patch_size (int): patch size for ViT.
        """
        super().__init__()

        self.model = torchvision.models.vit_b_16(
            image_size=image_size,
            patch_size=patch_size,
        )

        if in_channels != 3:
            old_conv = self.model.conv_proj
            self.model.conv_proj = nn.Conv2d(
                in_channels=in_channels,
                out_channels=old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias is not None,
            )

        self.model.heads.head = nn.Linear(
            self.model.heads.head.in_features, n_class
        )

    def forward(self, img, **batch):
        """
        Model forward method.

        Args:
            img (Tensor): input image tensor.
        Returns:
            output (dict): output dict containing logits.
        """
        return {"logits": self.model(img)}

    def __str__(self):
        """
        Model prints with the number of parameters.
        """
        all_parameters = sum(p.numel() for p in self.parameters())
        trainable_parameters = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )

        result_info = super().__str__()
        result_info += f"\nAll parameters: {all_parameters}"
        result_info += f"\nTrainable parameters: {trainable_parameters}"

        return result_info
