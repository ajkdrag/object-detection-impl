from einops import rearrange
from einops.layers.torch import Rearrange
from torch import nn

from ...utils.ml import swap_dims
from .layers import (
    ExpansionFCLayer,
    FlattenLayer,
    ScaledResidual,
)


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
        self.token_mixer = MLPTokenMixerBlock(
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
