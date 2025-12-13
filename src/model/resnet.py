import torch
import torchvision
from torch import nn


class ResNetModel(nn.Module):
    """
    Wrapper over ResNet18 from torchvision
    """

    def __init__(
            self,
            n_class: int,
            in_channels: int = 3,
            subsampling_kernel: int = 3,
            subsampling_stride: int = 1,
            disable_subsampling_max_pool: bool = False,
        ):
        """
        Args:
            n_class (int): number of classes.
            subsampling_kernel (int): kernel size of subsampling layer in resnet
            subsampling_stride (int): stride of subsampling layer in resnet 
        """
        super().__init__()

        self.model = torchvision.models.resnet18()
        
        if disable_subsampling_max_pool:
            self.model.maxpool = nn.Identity()

        self.model.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=self.model.conv1.out_channels,
            kernel_size=subsampling_kernel,
            stride=subsampling_stride,
            padding=(subsampling_kernel - 1) // 2,
        )

        self.model.fc = nn.Linear(self.model.fc.in_features, n_class)

    def forward(self, img, **batch):
        """
        Model forward method.

        Args:
            img (Tensor): input img.
        Returns:
            output (dict): output dict containing logits.
        """
        return {"logits": self.model(img)}

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

class ResNetImg32(nn.Module):
    """
    Pure Encoder: ResNet18 adapted for 32x32 images.
    This is both CIFAR100 and MNIST-ON-CIFAR10 (becasue cifar 10 is 32x32)
    Returns: Raw Layout (512 features).
    """
    def __init__(self, in_channels: int = 3):
        super().__init__()
        
        self.model = torchvision.models.resnet18(weights=None)
        
        # Adapt for (32x32)
        self.model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.maxpool = nn.Identity()
        self.model.fc = nn.Identity()

    def forward(self, x):
        # here not a dict because it's only a component in stacked - Boboxa, please check
        return torch.flatten(self.model(x), 1)