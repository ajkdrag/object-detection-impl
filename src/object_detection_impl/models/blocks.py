import torch
import torch.nn as nn
from object_detection_impl.models.activations import Activations


class BasicFCBlock(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        activation=Activations.RELU,
        dropout=0.2,
        bn=True,
    ):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_features, out_features, bias=not bn),
            nn.BatchNorm1d(out_features) if bn else nn.Identity(),
            activation.value(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x):
        return self.block(x)


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        padding="same",
        activation=Activations.RELU,
        pre=False,
        **kwargs,
    ):
        super().__init__()
        bn_act_block = nn.Sequential(
            nn.BatchNorm2d(in_channels) if pre else nn.BatchNorm2d(
                out_channels),
            activation.value(),
        )
        layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=False,
                **kwargs,
            ),
            bn_act_block,
        ]
        if pre:
            layers = layers.reverse()
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class Conv1x1Block(ConvBlock):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        padding="same",
        activation=Activations.RELU,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size=1,
            padding=padding,
            stride=stride,
            activation=activation,
        )

    def forward(self, x):
        return super().forward(x)


class Downsample1x1Block(Conv1x1Block):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride=2,
    ):
        super().__init__(
            in_channels,
            out_channels,
            stride=stride,
            padding=0,
            activation=Activations.ID,
        )

    def forward(self, x):
        return super().forward(x)


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        padding="same",
    ):
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(
                in_channels,
                out_channels,
                stride=stride,
                padding=padding,
            ),
            ConvBlock(
                out_channels,
                out_channels,
                activation=Activations.ID,
            ),
        )
        self.downsample = nn.Identity()
        if stride != 1:
            self.downsample = Downsample1x1Block(
                in_channels,
                out_channels,
                stride,
            )
        elif in_channels != out_channels:
            self.downsample = Conv1x1Block(
                in_channels,
                out_channels,
            )
        self.activation = Activations.RELU.value()

    def forward(self, x):
        skip_out = self.downsample(x)
        return self.activation(self.block(x) + skip_out)


class BottleneckBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        red_channels,
        out_channels=None,
        kernel_size=3,
        last_conv1x1=False,
    ):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels

        layers = [
            Conv1x1Block(in_channels, red_channels),
            ConvBlock(
                red_channels,
                red_channels if last_conv1x1 else out_channels,
                kernel_size,
            ),
        ]
        if last_conv1x1:
            layers.append(Conv1x1Block(red_channels, out_channels))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class InceptionBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        l_red_channels,
        l_out_channels,
        l_kernel_sizes,
        last_conv1x1=False,
    ):
        """
        l_red_channels: list of reduction channels
        l_out_channels: list of output channels
        l_kernel_sizes: list of kernel sizes
        """
        super().__init__()
        self.branches = nn.ModuleList([])
        for i, (red, out, kernel_size) in enumerate(
            zip(l_red_channels, l_out_channels, l_kernel_sizes)
        ):
            if kernel_size == 1:
                layers = (
                    []
                    if red is None
                    else [
                        nn.MaxPool2d(3, 1, 1),
                    ]
                )
                layers.append(Conv1x1Block(in_channels, out))
                branch = nn.Sequential(*layers)
            else:
                branch = BottleneckBlock(
                    in_channels,
                    red,
                    out,
                    kernel_size,
                    last_conv1x1=last_conv1x1,
                )
            self.branches.add_module(f"branch{i}", branch)

    def forward(self, x):
        return torch.cat([branch(x) for branch in self.branches], dim=1)
