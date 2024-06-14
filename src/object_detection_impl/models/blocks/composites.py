import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange
from object_detection_impl.models.blocks.activations import Activations
from object_detection_impl.models.blocks.attentions import Attentions
from object_detection_impl.models.blocks.core import (
    BottleneckLayer,
    Conv1x1Layer,
    ConvLayer,
    DenseShortcutLayer,
    ExpansionFCLayer,
    ShortcutLayer,
)
from object_detection_impl.utils.ml import swap_dims


class DenseNetBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        growth_rate=32,
        num_layers=3,
        activation=Activations.RELU(inplace=True),
    ):
        super().__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(DenseShortcutLayer(in_channels, growth_rate))
            in_channels += growth_rate
        self.block = nn.Sequential(*layers)
        self.out_channels = in_channels

    def forward(self, x):
        return self.block(x)


class ResNetBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels=None,
        stride=1,
        activation=Activations.RELU(inplace=True),
        attn: Attentions = Attentions.ID(),
    ):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels

        self.block = nn.Sequential(
            ConvLayer(
                in_channels,
                out_channels,
                stride=stride,
                kernel_size=3,
                padding=1,
                activation=activation,
            ),
            ConvLayer(
                out_channels,
                out_channels,
                activation=Activations.ID(),
            ),
        )
        self.shortcut = ShortcutLayer(in_channels, out_channels, stride)
        self.activation = activation.create()
        self.attn_block = attn.create(in_channels=out_channels)

    def forward(self, x):
        skip_out = self.shortcut(x)
        return self.activation(
            self.attn_block(self.block(x)) + skip_out,
        )


class ResNeXtBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        red_channels,
        out_channels=None,
        groups=1,
        stride=1,
        activation=Activations.RELU(inplace=True),
        attn: Attentions = Attentions.ID(),
    ):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels

        self.block = BottleneckLayer(
            in_channels,
            red_channels,
            out_channels,
            groups=groups,
            stride=stride,
            activation=activation,
            last_act=False,
        )
        self.shortcut = ShortcutLayer(in_channels, out_channels, stride)
        self.activation = activation.create()
        self.attn_block = attn.create(in_channels=out_channels)

    def forward(self, x):
        skip_out = self.shortcut(x)
        return self.activation(
            self.attn_block(self.block(x)) + skip_out,
        )


class ConvMixerBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        kernel_size=7,
        stride=1,
        padding="same",
        activation=Activations.GELU(),
    ):
        super().__init__()
        self.dw_conv = ConvLayer(
            in_channels,
            in_channels,
            kernel_size,
            groups=in_channels,
            activation=activation,
        )
        self.pw_conv = Conv1x1Layer(
            in_channels,
            in_channels,
            activation=activation,
        )
        self.shortcut = ShortcutLayer(in_channels, in_channels, stride)

    def forward(self, x):
        return self.pw_conv(self.shortcut(x) + self.dw_conv(x))


class MLPTokenMixerBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        num_patches,
        expansion=4,
        dropout=0.0,
        activation=Activations.GELU(),
    ):
        super().__init__()
        # [n, (h*w), c]
        self.block = nn.Sequential(
            nn.LayerNorm(in_channels),
            Rearrange("n p c -> n c p"),
            ExpansionFCLayer(
                num_patches,
                expansion,
                activation=activation,
                dropout=dropout,
                bn=False,
            ),
            Rearrange("n c p -> n p c"),
        )

    def forward(self, x):
        _, _, h, w = x.shape
        x = rearrange(x, "n c h w -> n (h w) c")
        x = x + self.block(x)
        return rearrange(x, "n (h w) c -> n c h w", h=h, w=w)


class ConvTokenMixerBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        num_patches,
        expansion=4,
        dropout=0.0,
        activation=Activations.GELU(),
    ):
        super().__init__()
        # [n, (h*w), c]
        self.block = nn.Sequential(
            nn.LayerNorm(in_channels),
            nn.Conv1d(num_patches, int(num_patches * expansion), 1),
            nn.Dropout(dropout),
            nn.Conv1d(int(num_patches * expansion), num_patches, 1),
            nn.Dropout(dropout),
            activation.create(),
        )

    def forward(self, x):
        _, _, h, w = x.shape
        x = rearrange(x, "n c h w -> n (h w) c")
        x = x + self.block(x)
        return rearrange(x, "n (h w) c -> n c h w", h=h, w=w)


class MLPDimMixerBlock(nn.Module):
    def __init__(
        self,
        norm_channels,
        mixing_channels,
        dim=-1,
        expansion=4,
        dropout=0.0,
        activation=Activations.GELU(),
    ):
        super().__init__()
        self.norm = nn.LayerNorm(norm_channels)
        self.dim = dim
        self.block = ExpansionFCLayer(
            mixing_channels,
            expansion,
            mixing_channels,
            activation,
            dropout,
            bn=False,
        )

    def forward(self, x):
        # x: [n, d1, d2, ... in_dims]
        y = self.norm(x)
        y = swap_dims(y, self.dim, -1)
        y = self.block(y)
        y = swap_dims(y, self.dim, -1)
        return x + y


class MLPChannelMixerBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        num_patches,
        expansion=0.5,
        dropout=0.0,
        activation=Activations.GELU(),
    ):
        super().__init__()
        # [n, (h*w), c]
        self.block = nn.Sequential(
            nn.LayerNorm(in_channels),
            ExpansionFCLayer(
                in_channels,
                expansion,
                in_channels,
                activation,
                dropout,
                bn=False,
            ),
        )

    def forward(self, x):
        _, _, h, w = x.shape
        x = rearrange(x, "n c h w -> n (h w) c")
        x = x + self.block(x)
        return rearrange(x, "n (h w) c -> n c h w", h=h, w=w)


class MLPMixerBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        num_patches,
        tm_expansion=8,
        cm_expansion=0.5,
        dropout=0.0,
        activation=Activations.GELU(),
    ):
        super().__init__()
        self.num_patches = num_patches
        self.token_mixer = ConvTokenMixerBlock(
            in_channels,
            num_patches,
            tm_expansion,
            dropout=dropout,
            activation=activation,
        )
        self.channel_mixer = MLPChannelMixerBlock(
            in_channels,
            num_patches,
            cm_expansion,
            dropout=dropout,
            activation=activation,
        )

    def forward(self, x):
        return self.channel_mixer(self.token_mixer(x))


class MDMLPMixerBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        norm_channels,
        patch_num_h,
        patch_num_w,
        expansion=4,
        dropout=0.0,
        activation=Activations.GELU(),
    ):
        super().__init__()
        self.num_patches_h = patch_num_h
        self.num_patches_w = patch_num_w
        l_dims = [2, 3, 1, 4]
        l_mixing_channels = [
            patch_num_h,
            patch_num_w,
            in_channels,
            norm_channels,
        ]
        self.block = nn.Sequential(
            *[
                MLPDimMixerBlock(
                    norm_channels=norm_channels,
                    mixing_channels=l_mixing_channels[i],
                    dim=l_dims[i],
                    expansion=expansion,
                    dropout=dropout,
                    activation=activation,
                )
                for i in range(len(l_dims))
            ],
        )

    def forward(self, x):
        # x: [n, c, p, d]
        # p = num_patches, d=patch_sz**2 (i.e. a flattened patch)
        x = rearrange(
            x,
            "n c (h w) d -> n c h w d",
            h=self.num_patches_h,
            w=self.num_patches_w,
        )
        return rearrange(self.block(x), "n c h w d -> n c (h w) d")


class InceptionBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        l_red_channels,
        l_out_channels,
        l_kernel_sizes,
        activation=Activations.RELU(inplace=True),
        last_conv1x1=False,
    ):
        """
        l_red_channels: list of reduction channels
        l_out_channels: list of output channels
        l_kernel_sizes: list of kernel sizes
        """
        super().__init__()
        self.branches = nn.ModuleList()
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
                layers.append(
                    Conv1x1Layer(
                        in_channels,
                        out,
                        activation=activation,
                    )
                )
                branch = nn.Sequential(*layers)
            else:
                branch = BottleneckLayer(
                    in_channels,
                    red,
                    out,
                    kernel_size,
                    padding="same",
                    last_conv1x1=last_conv1x1,
                    activation=activation,
                )
            self.branches.append(branch)

    def forward(self, x):
        return torch.cat([branch(x) for branch in self.branches], dim=1)

