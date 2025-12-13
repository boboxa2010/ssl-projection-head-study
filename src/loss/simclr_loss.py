import torch
from torch import nn
import torch.nn.functional as F

class SimCLRLoss(nn.Module):
    """
    Implementation of NT-Xent (Normalized Temperature-scaled Cross Entropy Loss)
    for Self-Supervised Learning (SimCLR).
    """
    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> dict:
        """
        Calculates the contrastive loss between two views of the batch.

        Args:
            z_i (Tensor): Projections of the first augmented view. 
                          Shape: [Batch_Size, Projection_Dim]
            z_j (Tensor): Projections of the second augmented view. 
                          Shape: [Batch_Size, Projection_Dim]

        Returns:
            losses (dict): dict containing calculated loss functions.
        """
        batch_size = z_i.shape[0]

        z_i_norm = F.normalize(z_i, dim=1)
        z_j_norm = F.normalize(z_j, dim=1)

        # Shape: [2*Batch_Size, Projection_Dim]
        z = torch.cat([z_i_norm, z_j_norm], dim=0)

        # Shape: [2*Batch_Size, 2*Batch_Size]
        similarity_matrix = torch.matmul(z, z.T)

        # Shape: [2*Batch_Size, 2*Batch_Size]
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
        
        similarity_matrix = similarity_matrix.masked_fill(mask, -float('inf'))

        # For index i (first view), the positive pair is i + batch_size (second view)
        # For index i + batch_size, the positive pair is i
        positives = torch.arange(batch_size, device=z.device)
        # Shape: [2*Batch_Size]
        labels = torch.cat([positives + batch_size, positives])

        similarity_matrix = similarity_matrix / self.temperature

        return {"loss": F.cross_entropy(similarity_matrix, labels)}
