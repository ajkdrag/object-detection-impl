import torch.nn as nn
from object_detection_impl.models.blocks import (
    ResidualBlock,
    BottleneckBlock,
    ConvBlock,
    InceptionBlock,
)
from object_detection_impl.utils.registry import load_obj
from omegaconf import DictConfig


class LeNet5(nn.Module):
    def __init__(self, config: DictConfig) -> None:
        super().__init__()

        self.layer1 = nn.Sequential(
            ConvBlock(3, 6, kernel_size=5, padding=0),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer2 = nn.Sequential(
            ConvBlock(6, 16, kernel_size=5, padding=0),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.head = load_obj(config.model.head.class_name)(
            in_features=16,
            **config.model.head.params,
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return self.head(x)


class LeNetV2(nn.Module):
    def __init__(self, config: DictConfig) -> None:
        super().__init__()
        mp = nn.MaxPool2d(kernel_size=2, stride=2)

        self.layer1 = nn.Sequential(
            ConvBlock(3, 6, kernel_size=5, padding=0),
            ConvBlock(6, 12, kernel_size=3),
            mp,
        )
        self.layer2 = nn.Sequential(
            InceptionBlock(
                12,
                [4, 3, 2],
                [8, 6, 4],
                [3, 5, 1],
                last_conv1x1=False,
            ),
            ResidualBlock(18, 10, stride=2),
        )

        self.head = load_obj(config.model.head.class_name)(
            in_features=10,
            **config.model.head.params,
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return self.head(x)
