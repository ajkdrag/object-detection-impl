import torch.nn as nn
from einops.layers.torch import Rearrange

from ..norm_factory import Norms
from .attentions import ConvLikeAttention, MultiHeadSA, MultiScaleSAV2
from .layers import (
    ApplyNorm,
    ExpansionFCLayer,
    LightBottleneckLayer,
    ScaledResidual,
    SoftSplit,
    UnflattenLayer,
)


class T2TBlock(nn.Sequential):
    def __init__(
        self,
        c1,
        c2,
        h,
        f=1,
        k=3,
        s=1,
        heads=1,
        act="gelu",
        drop=0.0,
    ):
        super().__init__(
            TransformerEncoder(c1, c1, f, heads, act, drop),
            UnflattenLayer(h=h),
            SoftSplit(c1, c2, k=k, s=s),
        )


class TransformerEncoderMultiScale(nn.Module):
    def __init__(
        self,
        c1,
        c2,
        h,
        f=2,
        k_q=1,
        k_kv=1,
        s_q=1,
        s_kv=1,
        heads=2,
        act="gelu",
        drop=0.0,
        pre_norm=True,
    ):
        super().__init__()
        msa = MultiScaleSAV2(
            c1,
            h,
            k_q,
            k_kv,
            s_q,
            s_kv,
            heads,
            drop,
        )
        msa_shortcut = nn.Sequential(
            Rearrange("n (h w) c -> n c h w", h=h),
            msa.pool_q,
        )

        mlp_shortcut = nn.Sequential(
            Rearrange("n p c -> n c p"),
            nn.Conv1d(c1, c2, 1),
            Rearrange("n c p -> n p c"),
        )

        self.block = nn.Sequential(
            ScaledResidual(
                ApplyNorm(Norms.get("ln", c1), msa, pre_norm),
                shortcut=msa_shortcut,
            ),
            ScaledResidual(
                ExpansionFCLayer(c1, c2, f, act, "ln", drop, pre_norm),
                shortcut=mlp_shortcut,
            ),
        )

    def forward(self, x):
        return self.block(x)


class TransformerEncoderConvLike(nn.Module):
    def __init__(
        self,
        c1,
        c2=None,
        k=3,
        s=1,
        f=2,
        heads=2,
        act="gelu",
        drop=0.0,
        pre_norm=True,
        **kwargs,
    ):
        super().__init__()
        c2 = c2 or c1
        self.block = nn.Sequential(
            ScaledResidual(
                ApplyNorm(
                    Norms.get("bn2d", c1),
                    ConvLikeAttention(c1, c2, s, heads=heads, drop=drop, **kwargs),
                    pre_norm,
                ),
                apply_shortcut=(s == 1 and c1 == c2),
            ),
            ScaledResidual(
                nn.BatchNorm2d(c2),
                LightBottleneckLayer(c2, k=k, f=f),
            ),
        )

    def forward(self, x):
        return self.block(x)


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        c1,
        c2=None,
        f=2,
        heads=2,
        act="gelu",
        drop=0.0,
        pre_norm=True,
        **kwargs,
    ):
        super().__init__()
        c2 = c2 or c1
        mlp_shortcut = nn.Sequential(
            Rearrange("n p c -> n c p"),
            nn.Conv1d(c1, c2, 1),
            Rearrange("n c p -> n p c"),
        )
        self.block = nn.Sequential(
            ScaledResidual(
                ApplyNorm(
                    Norms.get("ln", c1),
                    MultiHeadSA(c1, heads, drop, **kwargs),
                    pre_norm,
                )
            ),
            ScaledResidual(
                ExpansionFCLayer(c1, c2, f, act, "ln", drop, pre_norm),
                shortcut=mlp_shortcut if c1 != c2 else None,
            ),
        )

    def forward(self, x):
        return self.block(x)  # [n, p, c]
