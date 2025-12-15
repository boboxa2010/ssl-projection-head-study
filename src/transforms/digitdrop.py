import torch
import torch.nn as nn
import random


class DropDigit(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
    
    def forward(self, img, digit, s=0.5):
        mask = None
        if img.dim() == 3:
            if random.random() > self.p:
                return img
            mnist_padded = torch.nn.functional.pad(digit, pad=(2, 2, 2, 2, 0, 0))
            mnist_padded = mnist_padded.repeat(3, 1, 1)

        if img.dim() == 4:
            mask = (torch.rand(img.shape[0]) < self.p).int() # с вероятностью p мы применяем аугментацию
            if not mask.any():
                return img
            mnist_padded = torch.nn.functional.pad(
                    digit, 
                    (2, 2, 2, 2, 0, 0), 
                    mode='constant', 
                    value=0
                ).repeat(1, 3, 1, 1)

        digit_area = (mnist_padded > 0).float()
        cifar_background = img * (1 - digit_area)
        img_foreground = img * digit_area
        cifar_foreground = (img_foreground - (1 - s) * mnist_padded) / (s + 1e-5)
        
        cifar_image = cifar_background + cifar_foreground
        if mask is not None:
            indices = torch.where(mask == 1) 
            cifar_image[indices] = img[indices]
        return cifar_image
        

