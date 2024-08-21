from typing import List

import torch.nn as nn

from .layers import (
    BasicFCLayer,
    Conv1x1Layer,
)


class ConvHead(nn.Module):
    def __init__(
        self,
        c1: int,
        cn: List[int],
        act="relu",
        last_act="noop",
        norm="bn2d",
    ):
        super().__init__()
        prev = c1
        layers = []
        last = len(cn) - 1
        for idx, c_ in enumerate(cn):
            layers.append(
                Conv1x1Layer(
                    prev,
                    c_,
                    act=act if idx != last else last_act,
                    norm=norm if idx != last else "noop",
                )
            )
            prev = c_
        layers.append(nn.Flatten())
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class FCHead(nn.Module):
    def __init__(
        self,
        c1: int,
        cn: List[int],
        act="relu",
        last_act="noop",
        norm="bn1d",
        drop: float = 0.0,
    ):
        super().__init__()
        prev = c1
        layers = []
        last = len(cn) - 1
        for idx, c_ in enumerate(cn):
            layers.append(
                BasicFCLayer(
                    prev,
                    c_,
                    act=act if idx != last else last_act,
                    drop=drop if idx != last else 0.0,
                    norm=norm if idx != last else "noop",
                ),
            )
            prev = c_

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)
