from object_detection_impl.models.blocks.core import NormAct
from torch import nn


class AvgPool(nn.Module):
    def __init__(
        self,
        in_channels,
        norm="bn2d",
        act="relu",
        dropout=0.0,
    ):
        super().__init__()
        self.block = nn.Sequential(
            NormAct(in_channels, norm, act),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.block(x)
