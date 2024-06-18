from typing import Literal

import torch
from einops import rearrange
from einops.layers.torch import Rearrange
from object_detection_impl.models.act_factory import Acts
from object_detection_impl.models.norm_factory import Norms
from torch import nn


class NormAct(nn.Sequential):
    def __init__(self, in_channels, norm="bn2d", act="relu"):
        super().__init__(Norms.get(norm, in_channels), Acts.get(act))


class ApplyNorm(nn.Sequential):
    def __init__(self, norm, fn, order="pre"):
        layers = [norm, fn] if order == "pre" else [fn, norm]
        super().__init__(*layers)


class BasicFCLayer(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        dropout=0.0,
        act="relu",
        norm="ln",
        norm_order: Literal["pre", "post"] = "pre",
    ):
        super().__init__()
        fc_layers = [
            nn.Linear(
                in_features,
                out_features,
            ),
            Acts.get(act),
            nn.Dropout(dropout),
        ]
        self.norm = norm

        norm_dims = in_features if norm_order == "pre" else out_features
        self.block = ApplyNorm(
            Norms.get(norm, norm_dims),
            nn.Sequential(*fc_layers),
            norm_order,
        )

    def forward(self, x):
        return self.block(x)


class ExpansionFCLayer(nn.Module):
    def __init__(
        self,
        in_features,
        expansion=4,
        out_features=None,
        act="relu",
        dropout=0.0,
        norm="ln",
        norm_order: Literal["pre", "post"] = "pre",
    ):
        super().__init__()
        out_features = out_features or in_features
        fc_layers = [
            BasicFCLayer(
                in_features,
                int(in_features * expansion),
                act=act,
                dropout=0.0,
                norm="noop",
            ),
            BasicFCLayer(
                int(in_features * expansion),
                out_features,
                act="noop",
                dropout=dropout,
                norm="noop",
            ),
        ]
        norm_dims = in_features if norm_order == "pre" else out_features
        self.block = ApplyNorm(
            Norms.get(norm, norm_dims),
            nn.Sequential(*fc_layers),
            norm_order,
        )

    def forward(self, x):
        return self.block(x)


class SoftSplit(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels=None,
        kernel_size=3,
        stride=1,
    ):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.block = nn.Sequential(
            nn.Unfold(kernel_size, stride=stride, padding=padding),
            Rearrange("n c p -> n p c"),
            nn.Linear(in_channels * kernel_size**2, out_channels)
            if out_channels is not None
            else nn.Identity(),
        )

    def forward(self, x):
        return self.block(x)


class FlattenLayer(nn.Module):
    def __init__(
        self,
        norm_dims=None,
        norm="noop",
        norm_order: Literal["pre", "post"] = "pre",
    ):
        super().__init__()
        self.block = ApplyNorm(
            Norms.get(norm, norm_dims),
            nn.Identity(),
            norm_order,
        )

    def forward(self, x):
        if len(x.shape) == 4:  # Input shape: [n, c, h, w]
            return self.block(rearrange(x, "n c h w -> n (h w) c"))
        elif len(x.shape) == 3:  # Input shape: [n, p, c]
            return self.block(x)
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
        act="relu",
        pre_act=False,
        norm="bn2d",
        **kwargs,
    ):
        super().__init__()
        norm_dims = in_channels if pre_act else out_channels
        if stride > 1 and padding == "same":
            padding = (kernel_size - 1) // 2
        layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                stride=stride,
                **kwargs,
            ),
            NormAct(norm_dims, norm, act),
        ]
        if pre_act:
            layers.reverse()
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class DownsampleConvLayer(ConvLayer):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=4,
        **kwargs,
    ):
        padding = (kernel_size - 1) // 2
        super().__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=2,
            padding=padding,
            **kwargs,
        )


class DownsamplePoolLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        pool="avg",
        **kwargs,
    ):
        super().__init__()
        self.conv = Conv1x1Layer(
            in_channels,
            out_channels,
            **kwargs,
        )
        if pool == "avg":
            self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        elif pool == "max":
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        else:
            raise ValueError(f"Unsupported pool type: {pool}")

    def forward(self, x):
        return self.pool(self.conv(x))


class Conv1x1Layer(ConvLayer):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        act="relu",
        **kwargs,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size=1,
            padding=0,
            stride=stride,
            act=act,
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
        **kwargs,
    ):
        super().__init__(
            in_channels,
            out_channels,
            stride=stride,
            act="noop",
            **kwargs,
        )

    def forward(self, x):
        return super().forward(x)


class ShortcutLayer(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        if stride != 1:
            self.shortcut = DownsampleConvLayer(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                act="noop",
            )
        elif in_channels != out_channels:
            self.shortcut = Conv1x1Layer(
                in_channels,
                out_channels,
                act="noop",
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
        groups=1,
        stride=1,
        act="relu",
        norm="bn2d",
        last_conv1x1=True,
        last_act=True,
    ):
        super().__init__()
        out_channels = out_channels or in_channels
        layers = [
            Conv1x1Layer(
                in_channels,
                red_channels,
                act=act,
                norm=norm,
            ),
            # xception trick: no act in depthwise conv
            ConvLayer(
                red_channels,
                red_channels if last_conv1x1 else out_channels,
                kernel_size,
                stride=stride,
                groups=groups,
                act=act if (last_conv1x1 ^ last_act) else "noop",
                norm=norm if (last_conv1x1 ^ last_act) else "noop",
            ),
        ]
        if last_conv1x1:
            layers.append(
                Conv1x1Layer(
                    red_channels,
                    out_channels,
                    act=act if last_act else "noop",
                    norm=norm if last_act else "noop",
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
        act="relu",
        norm="bn2d",
    ):
        super().__init__()
        self.block = ConvLayer(
            in_channels,
            growth_rate,
            act=act,
            norm=norm,
            pre_act=True,
        )

    def forward(self, x):
        out = self.block(x)
        return torch.cat([x, out], dim=1)


class ScaledResidual(nn.Module):
    def __init__(self, *layers, shortcut=None):
        super().__init__()
        self.shortcut = nn.Identity() if shortcut is None else shortcut
        self.residual = nn.Sequential(*layers)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return self.shortcut(x) + self.gamma * self.residual(x)
