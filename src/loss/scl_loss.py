import torch
import torch.nn as nn
import torch.nn.functional as F

class SupConLoss(nn.Module):
    """
    Supervised Contrastive Learning Loss.
    Accepts z_i, z_j and labels. 
    Internally concatenates them to form the contrastive matrix (i want to make universal code).
    """
    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor, labels: torch.Tensor, **batch) -> dict:
        """
        Calculates the contrastive loss between two views of the batch.

        Args:
            z_i (Tensor): Projections of the first augmented view. 
                          Shape: [Batch_Size, Projection_Dim]
            z_j (Tensor): Projections of the second augmented view. 
                          Shape: [Batch_Size, Projection_Dim]
            labels (Tensor): Labels

        Returns:
            losses (dict): dict containing calculated loss functions.
        """
        batch_size = z_i.shape[0]

        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)

        # Shape: [2*Batch_Size, Projection_Dim]
        features = torch.cat([z_i, z_j], dim=0)
        # Shape: [2*Batch_Size]
        labels = torch.cat([labels, labels], dim=0)

        # Shape: [2*Batch_Size, 2*Batch_Size]
        similarity_matrix = torch.matmul(features, features.T) / self.temperature

        # labels.view(-1, 1) == labels.view(1, -1) creates a matrix where (i, j) is True 
        # if label[i] == label[j].
        labels_mask = torch.eq(labels.view(-1, 1), labels.view(1, -1)).float()

        mask_eye = torch.eye(features.shape[0], device=z_i.device)
        
        # Remove diagonal from the positive mask
        labels_mask = labels_mask - mask_eye 

        logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
        logits = similarity_matrix - logits_max.detach()

        exp_logits = torch.exp(logits) * (1 - mask_eye)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)

        # Sum log_probs only where mask == 1 (positives), then average by number of positives
        mean_log_prob_pos = (labels_mask * log_prob).sum(1) / (labels_mask.sum(1) + 1e-6)

        return {"loss": -mean_log_prob_pos.mean()}
