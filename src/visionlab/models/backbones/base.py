from typing import Any

import torch.nn as nn


class Backbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Any) -> Any:
        return x
