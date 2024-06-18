from typing import List

import torch.nn as nn
from object_detection_impl.models.blocks.core import BasicFCLayer
from object_detection_impl.models.heads.base import Head


class FullyConnectedHead(Head):
    def __init__(
        self,
        in_channels: int,
        layer_units: List[int],
        act="relu",
        last_act="noop",
        norm="bn1d",
        dropout: float = 0.0,
    ):
        super().__init__()
        prev_units = in_channels
        layers = []
        last = len(layer_units) - 1
        for idx, units in enumerate(layer_units):
            layers.append(
                BasicFCLayer(
                    prev_units,
                    units,
                    act=act if idx != last else last_act,
                    dropout=dropout if idx != last else 0.0,
                    norm=norm if idx != last else "noop",
                ),
            )
            prev_units = units

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)
