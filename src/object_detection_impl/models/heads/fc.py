from typing import List

import torch.nn as nn
from object_detection_impl.models.activations import Activations
from object_detection_impl.models.blocks import BasicFCBlock
from object_detection_impl.models.heads.base import Head


class FullyConnectedHead(Head):
    def __init__(
        self,
        in_features: int,
        layer_units: List[int],
        activations: List[str],
        dropout_rates: List[float],
        avg_pool_sz: int = 7,
    ):
        super().__init__()
        assert (
            len(layer_units) == len(activations) == len(dropout_rates)
        ), "Mismatch in lengths of layer_units, activations, and dropout_rates"

        self.pool = nn.AdaptiveAvgPool2d(output_size=avg_pool_sz)
        prev_units = in_features * avg_pool_sz * avg_pool_sz

        layers = []
        last = len(layer_units) - 1
        for idx, (units, activation, dropout_rate) in enumerate(
            zip(layer_units, activations, dropout_rates)
        ):
            layers.append(
                BasicFCBlock(
                    prev_units,
                    units,
                    self._get_activation(activation),
                    dropout_rate if idx != last else 0.0,
                    bn=True if idx != last else False,
                ),
            )
            prev_units = units

        self.fc_layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.pool(x)
        return self.fc_layers(x.view(x.size(0), -1))

    def _get_activation(self, name: str):
        return Activations[name.upper()]
