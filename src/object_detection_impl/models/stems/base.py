from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseStem(ABC, nn.Module):
    @abstractmethod
    def forward(self, x):
        pass

    def out_shape(self, c=3, h=32, w=32):
        return self.forward(torch.zeros((1, c, h, w))).shape
