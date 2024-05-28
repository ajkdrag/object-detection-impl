from enum import Enum
from functools import partial

import torch.nn as nn


class Activations(Enum):
    RELU = partial(nn.ReLU, inplace=True)
    TANH = nn.Tanh
    SIGMOID = nn.Sigmoid
    ID = nn.Identity
