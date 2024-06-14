import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange
from object_detection_impl.models.blocks.activations import Activations
from object_detection_impl.models.blocks.attentions import MultiHead_SA
from object_detection_impl.models.blocks.core import (
    BasicFCLayer,
    ConvLayer,
    UnflattenLayer,
)
from object_detection_impl.models.stems.base import BaseStem


class T2TPatch(BaseStem):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_sizes,
        strides,
        activation=Activations.ID(),
        **kwargs,
    ):
        super().__init__()
        layers = []
        last = len(strides) - 1
        for i, (k_sz, stride) in enumerate(zip(kernel_sizes, strides)):
            in_channels *= k_sz**2
            layers.extend(
                [
                    UnflattenLayer(),
                    nn.Unfold(k_sz, stride=stride, padding=stride // 2),
                    Rearrange("n c p -> n p c"),
                    MultiHead_SA(
                        in_channels,
                        num_heads=1,
                        expansion=1,
                    )
                    if i != last
                    else nn.Identity(),
                ]
            )
        # layers.append(nn.LayerNorm(in_channels))
        layers.append(nn.Linear(in_channels, out_channels))
        layers.append(UnflattenLayer())
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
        activation=Activations.ID(),
        bn=False,
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
                activation,
                dropout=0.0,
                bn=bn,
                flatten=False,
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
        activation=Activations.ID(),
        bn=False,
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
                        activation=activation,
                        bn=bn,
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
        activation=Activations.ID(),
        bn=False,
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
                activation,
                dropout=0.0,
                bn=bn,
                flatten=False,
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
