from enum import Enum

import torch.nn as nn


class Activations(Enum):
    GELU = nn.GELU
    RELU = nn.ReLU
    TANH = nn.Tanh
    SIGMOID = nn.Sigmoid
    ID = nn.Identity

    def __call__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        return self

    def create(self, **kwargs):
        local_kwargs = self.kwargs.copy()
        local_kwargs.update(kwargs)
        return self.value(*self.args, **local_kwargs)
