import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange
from object_detection_impl.models.act_factory import Acts
from object_detection_impl.models.attn_factory import Attentions
from object_detection_impl.models.blocks.attentions import MultiHeadSA
from object_detection_impl.models.blocks.core import (
    BottleneckLayer,
    Conv1x1Layer,
    ConvLayer,
    DenseShortcutLayer,
    ExpansionFCLayer,
    FlattenLayer,
    ScaledResidual,
    ShortcutLayer,
    SoftSplit,
    UnflattenLayer,
)
from object_detection_impl.utils.ml import swap_dims


class T2TBlock(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        image_h,
        num_heads=1,
        expansion=1,
        kernel_size=3,
        stride=1,
        dropout=0.0,
        act="gelu",
    ):
        super().__init__(
            TransformerEncoder(
                in_channels,
                num_heads,
                dropout,
                expansion,
                act,
            ),
            UnflattenLayer(h=image_h),
            SoftSplit(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
            ),
        )


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        in_channels,
        num_heads=2,
        dropout=0.0,
        expansion=4,
        act="gelu",
    ):
        super().__init__()
        self.block = nn.Sequential(
            ScaledResidual(
                MultiHeadSA(
                    in_channels,
                    num_heads,
                    dropout,
                    norm="ln",
                ),
            ),
            ScaledResidual(
                ExpansionFCLayer(
                    in_channels,
                    expansion,
                    dropout=dropout,
                    act=act,
                    norm="ln",
                    norm_order="pre",
                ),
            ),
        )

    def forward(self, x):
        out = self.block(x)  # [n, p, c]
        if len(x.shape) == 4:
            return rearrange(out, "n (h w) c -> n c h w", w=x.shape[-1])
        return out


class DenseNetBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        growth_rate=32,
        num_layers=3,
        act="relu",
        norm="bn2d",
    ):
        super().__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(
                DenseShortcutLayer(
                    in_channels,
                    growth_rate,
                    act=act,
                    norm=norm,
                )
            )
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
        kernel_size=3,
        stride=1,
        act="relu",
        norm="bn2d",
        attn="noop",
    ):
        super().__init__()
        out_channels = out_channels or in_channels
        layers = [
            ConvLayer(
                in_channels,
                out_channels,
                stride=stride,
                kernel_size=3,
                act=act,
                norm=norm,
            ),
            ConvLayer(
                out_channels,
                out_channels,
                act="noop",
                norm="noop",
            ),
            Attentions.get(attn, out_channels),
        ]

        self.block = nn.Sequential(
            ScaledResidual(
                *layers,
                shortcut=ShortcutLayer(in_channels, out_channels, stride),
            ),
            Acts.get(act),
        )

    def forward(self, x):
        return self.block(x)


class ResNeXtBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        red_channels,
        out_channels=None,
        groups=1,
        kernel_size=3,
        stride=1,
        act="relu",
        norm="bn2d",
        attn="noop",
    ):
        super().__init__()
        out_channels = out_channels or in_channels

        layers = [
            BottleneckLayer(
                in_channels,
                red_channels,
                out_channels,
                kernel_size=kernel_size,
                groups=groups,
                stride=stride,
                act=act,
                norm=norm,
                last_act=False,
            ),
            Attentions.get(attn, out_channels),
        ]

        self.block = nn.Sequential(
            ScaledResidual(
                *layers,
                shortcut=ShortcutLayer(in_channels, out_channels, stride),
            ),
            Acts.get(act),
        )

    def forward(self, x):
        return self.block(x)


class ConvMixerBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        kernel_size=7,
        stride=1,
        padding="same",
        act="gelu",
    ):
        super().__init__()
        self.block = nn.Sequential(
            ScaledResidual(
                ConvLayer(
                    in_channels,
                    in_channels,
                    kernel_size,
                    groups=in_channels,
                    act=act,
                ),
                ShortcutLayer(in_channels, in_channels, stride),
            ),
            Conv1x1Layer(
                in_channels,
                in_channels,
                act=act,
            ),
        )

    def forward(self, x):
        return self.block(x)


class MLPTokenMixerBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        num_patches,
        expansion=4,
        dropout=0.0,
        act="gelu",
    ):
        super().__init__()
        self.block = ScaledResidual(
            FlattenLayer(
                norm_dims=in_channels,
                norm="ln",
                norm_order="pre",
            ),
            Rearrange("n p c -> n c p"),
            ExpansionFCLayer(
                num_patches,
                expansion,
                act=act,
                dropout=dropout,
                norm="noop",
            ),
            Rearrange("n c p -> n p c"),
        )

    def forward(self, x):
        return rearrange(
            self.block(x),
            "n (h w) c -> n c h w",
            w=x.shape[-1],
        )


class ConvTokenMixerBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        num_patches,
        expansion=4,
        dropout=0.0,
        act="gelu",
    ):
        super().__init__()
        self.block = ScaledResidual(
            FlattenLayer(
                norm_dims=in_channels,
                norm="ln",
                norm_order="pre",
            ),
            nn.Conv1d(num_patches, int(num_patches * expansion), 1),
            nn.Dropout(dropout),
            nn.Conv1d(int(num_patches * expansion), num_patches, 1),
            nn.Dropout(dropout),
            Acts.get(act),
        )

    def forward(self, x):
        return rearrange(
            self.block(x),
            "n (h w) c -> n c h w",
            w=x.shape[-1],
        )


class MLPDimMixerBlock(nn.Module):
    def __init__(
        self,
        norm_channels,
        mixing_channels,
        dim=-1,
        expansion=4,
        dropout=0.0,
        act="gelu",
    ):
        super().__init__()
        self.norm = nn.LayerNorm(norm_channels)
        self.dim = dim
        self.block = ExpansionFCLayer(
            mixing_channels,
            expansion,
            act=act,
            dropout=dropout,
            norm="noop",
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
        expansion=0.5,
        dropout=0.0,
        act="gelu",
    ):
        super().__init__()
        self.block = ScaledResidual(
            FlattenLayer(
                norm_dims=in_channels,
                norm="ln",
                norm_order="pre",
            ),
            ExpansionFCLayer(
                in_channels,
                expansion,
                act=act,
                dropout=dropout,
                norm="noop",
            ),
        )

    def forward(self, x):
        return rearrange(
            self.block(x),
            "n (h w) c -> n c h w",
            w=x.shape[-1],
        )


class MLPMixerBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        num_patches,
        tm_expansion=4,
        cm_expansion=0.5,
        dropout=0.0,
        act="gelu",
    ):
        super().__init__()
        self.num_patches = num_patches
        self.token_mixer = ConvTokenMixerBlock(
            in_channels,
            num_patches,
            tm_expansion,
            dropout=dropout,
            act=act,
        )
        self.channel_mixer = MLPChannelMixerBlock(
            in_channels,
            cm_expansion,
            dropout=dropout,
            act=act,
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
        act="gelu",
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
                    activation=act,
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
        act="relu",
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
                        act=act,
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
                    act=act,
                )
            self.branches.append(branch)

    def forward(self, x):
        return torch.cat([branch(x) for branch in self.branches], dim=1)
