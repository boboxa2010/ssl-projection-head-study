import torch
import torchvision
from torch import nn

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
        return torch.flatten(self.model(x), 1)