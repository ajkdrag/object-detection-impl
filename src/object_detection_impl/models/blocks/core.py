import torch
from einops import rearrange
from object_detection_impl.models.blocks.activations import Activations
from torch import nn


class BasicFCLayer(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        activation=Activations.RELU(),
        dropout=0.2,
        bn=True,
        flatten=True,
    ):
        super().__init__()
        layers = [nn.Flatten()] if flatten else []
        layers.append(
            nn.Linear(
                in_features,
                out_features,
                bias=not bn,
            )
        )
        if bn:
            layers.append(nn.BatchNorm1d(out_features))
        layers.append(activation.create())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class ExpansionFCLayer(nn.Sequential):
    def __init__(
        self,
        in_features,
        expansion=4,
        out_features=None,
        activation=Activations.RELU(),
        dropout=0.2,
        bn=True,
    ):
        super().__init__(
            BasicFCLayer(
                in_features,
                int(in_features * expansion),
                activation=activation,
                dropout=dropout,
                flatten=False,
                bn=bn,
            ),
            BasicFCLayer(
                int(in_features * expansion),
                in_features if out_features is None else out_features,
                activation=Activations.ID(),
                flatten=False,
                dropout=0.0,
                bn=False,
            ),
        )


class FlattenLayer(nn.Module):
    def forward(self, x):
        if len(x.shape) == 4:  # Input shape: [n, c, h, w]
            return rearrange(x, "n c h w -> n (h w) c")
        elif len(x.shape) == 3:  # Input shape: [n, p, c]
            return x
        else:
            raise ValueError(f"Unsupported input shape: {x.shape}")


class UnflattenLayer(nn.Module):
    def __init__(self, h=0):
        super().__init__()
        self.h = h

    def forward(self, x):
        if len(x.shape) == 4:  # Input shape: [n, c, h, w]
            return x
        elif len(x.shape) == 3:  # [n, p, c]
            p = x.shape[1]
            self.h = self.h or int(p**0.5)
            return rearrange(x, "n (h w) c -> n c h w", h=self.h)
        else:
            raise ValueError(f"Unsupported input shape: {x.shape}")


class ConvLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        padding="same",
        stride=1,
        activation=Activations.RELU(),
        pre=False,
        bn=True,
        **kwargs,
    ):
        super().__init__()
        if bn:
            bn_act_layers = [
                nn.BatchNorm2d(in_channels) if pre else nn.BatchNorm2d(
                    out_channels),
                activation.create(),
            ]
        else:
            bn_act_layers = [activation.create()]

        bn_act_block = nn.Sequential(*bn_act_layers)
        layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=not bn or pre,
                stride=stride,
                **kwargs,
            ),
            bn_act_block,
        ]
        if pre:
            layers.reverse()
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class Conv1x1Layer(ConvLayer):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        activation=Activations.RELU(inplace=True),
        **kwargs,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size=1,
            padding=0,
            stride=stride,
            activation=activation,
            **kwargs,
        )

    def forward(self, x):
        return super().forward(x)


class Downsample1x1Layer(Conv1x1Layer):
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
            activation=Activations.ID(),
        )

    def forward(self, x):
        return super().forward(x)


class ShortcutLayer(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        if stride != 1:
            self.shortcut = Downsample1x1Layer(
                in_channels,
                out_channels,
                stride,
            )
        elif in_channels != out_channels:
            self.shortcut = Conv1x1Layer(
                in_channels,
                out_channels,
                activation=Activations.ID(),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        return self.shortcut(x)


class BottleneckLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        red_channels,
        out_channels=None,
        kernel_size=3,
        padding=1,
        groups=1,
        stride=1,
        activation=Activations.RELU(inplace=True),
        last_conv1x1=True,
        last_act=True,
    ):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels

        layers = [
            Conv1x1Layer(
                in_channels,
                red_channels,
                activation=activation,
            ),
            # xception trick: no act in depthwise conv
            ConvLayer(
                red_channels,
                red_channels if last_conv1x1 else out_channels,
                kernel_size,
                padding=padding,
                stride=stride,
                groups=groups,
                activation=activation
                if (last_conv1x1 ^ last_act)
                else Activations.ID(),
            ),
        ]
        if last_conv1x1:
            layers.append(
                Conv1x1Layer(
                    red_channels,
                    out_channels,
                    activation=activation if last_act else Activations.ID(),
                )
            )
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class DenseShortcutLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        growth_rate=32,
        activation=Activations.RELU(inplace=True),
    ):
        super().__init__()
        self.block = ConvLayer(
            in_channels,
            growth_rate,
            pre=True,
        )

    def forward(self, x):
        out = self.block(x)
        return torch.cat([x, out], dim=1)
