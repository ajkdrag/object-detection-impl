import torch.nn as nn
from visionlab.models.blocks.core import (
    ConvLayer,
    ScaledResidual,
)
from visionlab.utils.ml import init_linear
from visionlab.utils.registry import load_obj
from omegaconf import DictConfig


class Block(ScaledResidual):
    def __init__(self, channels, kernel_size=3, stride=1, mult=4):
        mid_channels = channels * mult
        kernel_size = kernel_size + stride - 1
        super().__init__(
            ConvLayer(
                channels,
                mid_channels,
                kernel_size,
                stride=stride,
                groups=channels,
                padding=(kernel_size - 1) // 2,
            ),
            ConvLayer(
                mid_channels,
                channels,
                1,
            ),
            shortcut=nn.AvgPool2d(stride) if stride > 1 else None,
        )


class Stage(nn.Sequential):
    def __init__(
        self,
        channels,
        num_blocks,
        kernel_size=3,
        stride=1,
        mult=4,
    ):
        super().__init__(
            Block(channels, kernel_size, stride, mult),
            *[Block(channels, kernel_size, 1, mult)
              for _ in range(num_blocks - 1)],
        )


class StageStack(nn.Sequential):
    def __init__(
        self,
        channels,
        num_blocks,
        strides,
        kernel_size=3,
        mult=4,
    ):
        super().__init__(
            *[
                Stage(channels, num_blocks, kernel_size, stride, mult)
                for stride in strides
            ]
        )


class ConvTiny(nn.Module):
    def __init__(self, config: DictConfig) -> None:
        super().__init__()
        embed_sz = 128
        # inter_channels = 48
        strides = [1, 2, 2, 2]

        self.stem = nn.Conv2d(3, embed_sz, 3, padding=1, bias=False)
        self.trunk = StageStack(embed_sz, 2, strides, 3, 4)
        self.neck = load_obj(config.model.neck.class_name)(
            in_channels=embed_sz,
            **config.model.neck.get("params", {}),
        )

        self.head = load_obj(config.model.head.class_name)(
            in_channels=embed_sz,
            **config.model.head.params,
        )

        self.apply(init_linear)

    def forward(self, x):
        return self.head(self.neck(self.trunk(self.stem(x))))
