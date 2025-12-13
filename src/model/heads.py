import torch
import torchvision
from torch import nn

class UniversalProjectionHead(nn.Module):
    """
    Realization of Projection Head with ability to choose between MLP and fixed reweighting head
    from https://arxiv.org/abs/2403.11391
    """

    def __init__(
        self, 
        input_dim: int = 512, 
        hidden_dim: int = 512, 
        output_dim: int = 128, # only in MLP
        mode: str = 'mlp',  # mlp or fixed
        kappa: float = None # only in fixed head
    ):
        super().__init__()
        self.mode = mode
        self.output_dim = output_dim

        if self.mode == 'mlp':
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim)
            )
        
        elif self.mode == 'fixed':
            if kappa is None:
                raise ValueError("Kappa is required for fixed reweighting mode!")
            
            # [1, 1/k, 1/k^2, ... 1/k^(p-1)]
            
            exponents = torch.arange(input_dim, dtype=torch.float32)
            scales = 1.0 / (kappa ** exponents)
            self.register_buffer('scales', scales)
            
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # here not a dict because it's only a component in stacked - Boboxa, please check
        if self.mode == 'mlp':
            # [Batch_size, 128]
            return self.net(x)
        
        elif self.mode == 'fixed':
            # [Batch_size, 512]
            return x * self.scales
