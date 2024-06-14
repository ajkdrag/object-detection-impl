from typing import List

import torch.nn as nn
from object_detection_impl.models.blocks.activations import Activations
from object_detection_impl.models.blocks.core import BasicFCLayer
from object_detection_impl.models.heads.base import Head


class FullyConnectedHead(Head):
    def __init__(
        self,
        in_features: int,
        layer_units: List[int],
        activations: List[str],
        dropout_rates: List[float],
    ):
        super().__init__()
        assert (
            len(layer_units) == len(activations) == len(dropout_rates)
        ), "Mismatch in lengths of layer_units, activations, and dropout_rates"

        prev_units = in_features
        layers = []
        last = len(layer_units) - 1
        for idx, (units, activation, dropout_rate) in enumerate(
            zip(layer_units, activations, dropout_rates)
        ):
            layers.append(
                BasicFCLayer(
                    prev_units,
                    units,
                    self._get_activation(activation),
                    dropout_rate if idx != last else 0.0,
                    bn=True if idx != last else False,
                ),
            )
            prev_units = units

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

    def _get_activation(self, name: str):
        return Activations[name.upper()]()
