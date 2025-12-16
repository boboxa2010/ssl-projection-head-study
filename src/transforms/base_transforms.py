from torch import nn, Tensor


class BatchSequential(nn.Sequential):
    """
    A Sequential container that supports batch-aware modules
    """

    def forward(self, x: Tensor, **batch):
        for module in self:
            try:
                x = module(x, **batch)
            except TypeError:
                x = module(x)
        return x