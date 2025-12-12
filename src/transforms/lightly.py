import torch
import torch.nn as nn
from torchvision.transforms.functional import to_pil_image


class LightlySingleView(nn.Module):
    def __init__(self, transform, view_index=0):
        super().__init__()
        self.transform = transform
        self.view_index = view_index
    
    def forward(self, x):
        if x.dim() == 4:
            batch_size = x.shape[0]
            transformed = []
            for i in range(batch_size):
                img_pil = to_pil_image(x[i])
                views = self.transform(img_pil)
                transformed.append(views[self.view_index].unsqueeze(0))
            return torch.cat(transformed, dim=0).to(x.device)
        else:
            img_pil = to_pil_image(x)
            views = self.transform(img_pil)
            return views[self.view_index].to(x.device)

