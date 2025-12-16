import torch
from torch import nn


class BarlowTwinsLoss(nn.Module):
    """
    Implementation of BarlowTwinsLoss.
    """

    def __init__(self, lambda_coef: float = 5e-3):
        super().__init__()
        self.lambda_coef = lambda_coef

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> dict:
        """
        Calculates the loss with cross-correlation matrix.
        Loss = sum(1 - C_{ii})^2 + lambda_coef * sum(C_{ij}^2)

        Args:
            z_i (Tensor): Projections of the first augmented view.
                          Shape: [Batch_Size, Projection_Dim]
            z_j (Tensor): Projections of the second augmented view.
                          Shape: [Batch_Size, Projection_Dim]

        Returns:
            losses (dict): dict containing calculated loss functions.
        """
        batch_size = z_i.shape[0]
        feature_dim = z_i.shape[1]

        z_i_norm = (z_i - z_i.mean(dim=0)) / (z_i.std(dim=0) + 1e-6)
        z_j_norm = (z_j - z_j.mean(dim=0)) / (z_j.std(dim=0) + 1e-6)

        c = torch.mm(z_i_norm.T, z_j_norm) / batch_size

        diag_sum = (torch.diagonal(c) - 1).pow(2).sum()

        mask = torch.eye(feature_dim, device=c.device).bool()
        off_diag_sum = c[~mask].pow(2).sum()

        loss = diag_sum + self.lambda_coef * off_diag_sum

        return {"loss": loss}
