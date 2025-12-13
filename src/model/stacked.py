import torch
import torchvision
from torch import nn
from .resnet import ResNetImg32
from .heads import UniversalProjectionHead
from .ViT import ViTImg32

class SimCLR(nn.Module):
    """
    SimCLR realization with ability to choose encoders and heads
    """

    def __init__(
        self, 
        encoder: str = 'ResNet32', # or ViT32
        mode: str = 'mlp',  # mlp or fixed
        kappa: float = None # only in fixed head
    ):
        super().__init__()
        if encoder == 'ResNet32':
            self.encoder = ResNetImg32()
            input_dim = 512
        elif encoder == 'ViT32':
            self.encoder = ViTImg32()
            input_dim = 768
            #output - 768 maybe change ViT inside more?
        else:
            raise ValueError(f"Unknown encoder: {encoder}")
        
        self.projection_head = UniversalProjectionHead(input_dim, input_dim, 128, mode, kappa)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x) # h
        projections = self.projection_head(features) # z
        
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

class ProjectedClassifier(nn.Module):
    """
    Model for Supervised Learning experiment (SL and SCL) described in the paper.
    Structure: 
    Backbone -> Projection Head -> Linear Classifier if SL
    Backbone -> Projection Head -> ContrastiveLoss if SCL
    """
    def __init__(
        self, 
        n_classes = None, # In CIFAR-100 20 super classes but None if SCL task
        encoder: str = 'ResNet32', # or ViT32
        mode: str = 'mlp',  # mlp or fixed
        kappa: float = None # only in fixed head
    ):
        super().__init__()
        
        if encoder == 'ResNet32':
            self.encoder = ResNetImg32()
            input_dim = 512
        elif encoder == 'ViT32':
            self.encoder = ViTImg32()
            #output - 768 maybe change ViT inside more?
            input_dim = 768
        else:
            raise ValueError(f"Unknown encoder: {encoder}")
        
        self.projection_head = UniversalProjectionHead(input_dim, input_dim, 128, mode, kappa)

        if n_classes:
            if mode == 'mlp':
                classifier_input = self.projection_head.output_dim # 128
            else:
                classifier_input = input_dim # 512 or 768 (fixed head keeps dim)
            
            self.classifier = nn.Linear(classifier_input, n_classes)
        
        else:
            self.classifier = None

    def forward(self, x: torch.Tensor, **batch) -> dict:
        features = self.encoder(x) # h
        projections = self.projection_head(features) # z
        
        output = {
            "features": features,
            "projections": projections
        }

        if self.classifier:
            logits = self.classifier(projections)
            output["logits"] = logits

        return output

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
