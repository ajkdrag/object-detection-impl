import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange
from object_detection_impl.models.blocks.composites import (
    T2TBlock,
)
from object_detection_impl.models.blocks.core import (
    BasicFCLayer,
    ConvLayer,
    SoftSplit,
    UnflattenLayer,
)
from object_detection_impl.models.stems.base import BaseStem


class T2TPatchV2(BaseStem):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_sizes,
        strides,
        image_h,
        expansion=0.5,
        act="gelu",
        dropout=0.0,
        **kwargs,
    ):
        super().__init__()
        layers = []
        last = len(strides) - 1
        inter_channels = int(expansion * out_channels)

        for i, (k_sz, stride) in enumerate(zip(kernel_sizes, strides)):
            layers.extend(
                [
                    SoftSplit(
                        in_channels,
                        out_channels if i == last else inter_channels,
                        k_sz,
                        stride,
                    )
                    if i == 0
                    else T2TBlock(
                        inter_channels,
                        out_channels if i == last else inter_channels,
                        image_h,
                        kernel_size=k_sz,
                        stride=stride,
                        num_heads=kwargs.get("mha_heads", 1),
                        expansion=kwargs.get("mha_expansion", 1),
                        act=act,
                        dropout=dropout,
                    ),
                ]
            )
            image_h //= stride
        layers.append(UnflattenLayer(h=image_h))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class T2TPatch(BaseStem):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_sizes,
        strides,
        image_h,
        act="gelu",
        dropout=0.0,
        **kwargs,
    ):
        super().__init__()
        layers = []
        last = len(strides) - 1
        inter_channels = in_channels
        for i, (k_sz, stride) in enumerate(zip(kernel_sizes, strides)):
            inter_channels *= k_sz**2
            layers.extend(
                [
                    SoftSplit(
                        in_channels,
                        out_channels if i == last else inter_channels,
                        k_sz,
                        stride,
                    )
                    if i == 0
                    else T2TBlock(
                        inter_channels,
                        out_channels if i == last else None,
                        image_h,
                        kernel_size=k_sz,
                        stride=stride,
                        num_heads=kwargs.get("mha_heads", 1),
                        expansion=kwargs.get("mha_expansion", 1),
                        act=act,
                        dropout=dropout,
                    ),
                ]
            )
            image_h //= stride
        layers.append(UnflattenLayer(h=image_h))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class MDMLPPatch(BaseStem):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        act="noop",
        norm="noop",
        dropout=0.0,
        **kwargs,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.block = nn.Sequential(
            nn.Unfold(kernel_size=kernel_size, stride=stride),
            Rearrange(
                "n (c d) p -> n c p d",
                c=in_channels,
                d=kernel_size**2,
            ),
            BasicFCLayer(
                kernel_size**2,
                out_channels,
                dropout=dropout,
                act=act,
                norm=norm,
                norm_order="post",
            ),
        )

    def forward(self, x):
        return self.block(x)


class ConvPatch(BaseStem):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        n_layers=2,
        expansion=0.5,
        act="noop",
        norm="noop",
        padding=0,
        mp=True,
        **kwargs,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        inter_channels = int(expansion * out_channels)
        self.block = nn.Sequential(
            *[
                nn.Sequential(
                    ConvLayer(
                        in_channels if i == 0 else inter_channels,
                        out_channels if i == n_layers - 1 else inter_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        act=act,
                        norm=norm,
                        padding=padding,
                    ),
                    nn.MaxPool2d(
                        kernel_size=kwargs.get("mp_ksize", 3),
                        stride=kwargs.get("mp_stride", 2),
                        padding=kwargs.get("mp_padding", 1),
                    )
                    if mp
                    else nn.Identity(),
                )
                for i in range(n_layers)
            ]
        )

    def forward(self, x):
        return self.block(x)


class FCPatch(BaseStem):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        input_size,
        dropout=0.0,
        act="noop",
        norm="noop",
        **kwargs,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.block = nn.Sequential(
            nn.Unfold(kernel_size=kernel_size, stride=stride),
            Rearrange("n c p -> n p c"),
            BasicFCLayer(
                kernel_size**2 * in_channels,
                out_channels,
                act=act,
                dropout=dropout,
                norm=norm,
                norm_order="post",
            ),
        )

    def forward(self, x):
        _, _, h, w = x.shape
        return rearrange(
            self.block(x),
            "n (h w) c -> n c h w",
            h=(h - self.kernel_size) // self.stride + 1,
            w=(w - self.kernel_size) // self.stride + 1,
        )
