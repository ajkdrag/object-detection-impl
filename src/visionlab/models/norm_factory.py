import torch
from torch import nn

noop = nn.Identity()


class GRN(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(2, 3), keepdim=True)
        Nx = Gx / (Gx.mean(dim=1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


class LayerNormChannels(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        # x: [n, c, h, w]
        x = x.transpose(1, -1)
        x = self.norm(x)
        x = x.transpose(-1, 1)
        return x


class Norms:
    @staticmethod
    def get(norm: str, *args, **kwargs):
        if norm == "noop":
            return noop
        if norm == "bn1d":
            return nn.BatchNorm1d(*args, **kwargs)
        elif norm == "bn2d":
            return nn.BatchNorm2d(*args, **kwargs)
        elif norm == "ln":
            return nn.LayerNorm(*args, **kwargs)
        elif norm == "lnch":
            return LayerNormChannels(*args, **kwargs)
        elif norm == "grn":
            return GRN(*args, **kwargs)
        else:
            raise NotImplementedError
