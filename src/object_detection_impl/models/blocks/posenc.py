import torch
from torch import nn


class LearnablePositionEnc(nn.Module):
    def __init__(self, sizes):
        super().__init__()
        self.pos_enc = nn.Parameter(torch.zeros(1, *sizes))
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Parameter):
            nn.init.trunc_normal_(m, std=0.2)

    def forward(self, x):
        return x + self.pos_enc
