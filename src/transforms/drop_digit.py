import torch
import torch.nn as nn
import random


class DropDigit(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
    
    def forward(self, x, digit, s=0.5, **batch):
        mask = None
        if x.dim() == 3:
            if torch.rand(()) > self.p:
                return x
            mnist_padded = torch.nn.functional.pad(digit, pad=(2, 2, 2, 2, 0, 0))
            mnist_padded = mnist_padded.repeat(3, 1, 1)

        if x.dim() == 4:
            mask = (torch.rand(x.shape[0]) < self.p).int() # с вероятностью p мы применяем аугментацию
            if not mask.any():
                return x
            mnist_padded = torch.nn.functional.pad(
                    digit, 
                    (2, 2, 2, 2, 0, 0), 
                    mode='constant', 
                    value=0
                ).repeat(1, 3, 1, 1)

        digit_area = (mnist_padded > 0).float()
        cifar_background = x * (1 - digit_area)
        img_foreground = x * digit_area
        cifar_foreground = (img_foreground - (1 - s) * mnist_padded) / (s + 1e-5)
        
        cifar_image = cifar_background + cifar_foreground
        if mask is not None:
            indices = torch.where(mask == 1) 
            cifar_image[indices] = x[indices]
        return cifar_image
        

