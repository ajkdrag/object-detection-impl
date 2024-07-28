import torch
import torch.nn as nn

from ...utils.ml import make_divisible
from ..act_factory import Acts
from ..attn_factory import Attentions
from .layers import (
    BottleneckLayer,
    Conv1x1Layer,
    ConvLayer,
    DenseShortcutLayer,
    DWConvLayer,
    DWLightBottleneckLayer,
    DWSepShortcutLayer,
    GhostLayer,
    LightBottleneckLayer,
    ScaledResidual,
    Shortcut3x3Layer,
    ShortcutLayer,
)


class MobileNetBlock(nn.Module):
    def __init__(
        self,
        c1,
        c2=None,
        f=2,
        k=3,
        s=1,
        act="relu",
        norm="bn2d",
        attn="noop",
    ):
        super().__init__()
        c2 = c2 or c1
        self.block = nn.Sequential(
            ScaledResidual(
                DWLightBottleneckLayer(c1, c2, f, k, s, act, norm),
                Attentions.get(attn, c2),
                shortcut=ShortcutLayer(c1, c2, s),
                skip=(s != 1 or c1 != c2),
            ),
            Acts.get(act),
        )

    def forward(self, x):
        return self.block(x)


class GhostBottleneckBlock(nn.Module):
    def __init__(
        self,
        c1,
        c2=None,
        f=3,
        k=3,
        s=1,
        act="relu",
        norm="bn2d",
        attn="noop",
    ):
        super().__init__()
        c2 = c2 or c1
        c_ = make_divisible(f * c1, 4)
        downsample = (
            nn.Identity()
            if s == 1
            else DWConvLayer(
                c_,
                c_,
                k,
                s,
                act="noop",
            )
        )
        self.block = ScaledResidual(
            GhostLayer(c1, c_, k=k, act=act, norm=norm),
            downsample,
            Attentions.get(attn, c_),
            GhostLayer(c_, c2, k=k, act="noop", norm=norm),
            shortcut=DWSepShortcutLayer(c1, c2, k, s, act, norm),
        )

    def forward(self, x):
        return self.block(x)


class StackedDenseBlock(nn.Module):
    def __init__(
        self,
        n,
        c1,
        f=32,
        act="relu",
        norm="bn2d",
    ):
        super().__init__()
        layers = []
        for _ in range(n):
            layers += [DenseShortcutLayer(c1, f, act=act, norm=norm)]
            c1 += f
        self.block = nn.Sequential(*layers)
        self.c2 = c1

    def forward(self, x):
        return self.block(x)


class LightResNetBlock(nn.Module):
    def __init__(
        self,
        c1,
        c2=None,
        k=3,
        s=1,
        act="relu",
        norm="bn2d",
        attn="noop",
    ):
        super().__init__()
        c2 = c2 or c1
        self.block = nn.Sequential(
            ScaledResidual(
                nn.MaxPool2d(2, s) if s > 1 else nn.Identity(),
                BottleneckLayer(c1, c2, k, 1, act, norm),
                Attentions.get(attn, c2),
                shortcut=ShortcutLayer(c1, c2, s),
            ),
        )

    def forward(self, x):
        return self.block(x)


class ResNetBlock(nn.Module):
    def __init__(
        self,
        c1,
        c2=None,
        k=3,
        s=1,
        act="relu",
        norm="bn2d",
        attn="noop",
    ):
        super().__init__()
        c2 = c2 or c1

        self.block = nn.Sequential(
            ScaledResidual(
                BottleneckLayer(c1, c2, k, s, act, norm),
                Attentions.get(attn, c2),
                shortcut=Shortcut3x3Layer(c1, c2, s),
            ),
            Acts.get(act),
        )

    def forward(self, x):
        return self.block(x)


class ResNeXtBlock(nn.Module):
    def __init__(
        self,
        c1,
        c2=None,
        f=0.5,
        k=3,
        s=1,
        g=1,
        act="relu",
        norm="bn2d",
        attn="noop",
    ):
        super().__init__()
        c2 = c2 or c1

        self.block = nn.Sequential(
            ScaledResidual(
                LightBottleneckLayer(c1, c2, f, k, s, g, act, norm),
                Attentions.get(attn, c2),
                shortcut=ShortcutLayer(c1, c2, s),
            ),
            Acts.get(act),
        )

    def forward(self, x):
        return self.block(x)


class ConvMixerBlock(nn.Module):
    def __init__(
        self,
        c1,
        k=7,
        s=1,
        act="gelu",
        norm="bn2d",
    ):
        super().__init__()
        self.block = nn.Sequential(
            ScaledResidual(
                ConvLayer(
                    c1,
                    c1,
                    k,
                    s,
                    g=c1,
                    act=act,
                    norm=norm,
                ),
                ShortcutLayer(c1, c1, s),
            ),
            Conv1x1Layer(c1, c1, act=act, norm=norm),
        )

    def forward(self, x):
        return self.block(x)


class ConvTokenMixerBlock(nn.Module):
    def __init__(
        self,
        c1,
        seq,
        f=4,
        act="gelu",
        drop=0.0,
    ):
        super().__init__()
        self.block = ScaledResidual(
            nn.LayerNorm(c1),
            nn.Conv1d(seq, int(seq * f), 1),
            nn.Dropout(drop),
            nn.Conv1d(int(seq * f), seq, 1),
            nn.Dropout(drop),
            Acts.get(act),
        )

    def forward(self, x):
        return self.block(x)  # [n, p, c]


class InceptionBlock(nn.Module):
    def __init__(
        self,
        c1,
        cn,
        fn,
        kn,
        act="relu",
        norm="bn2d",
        last_conv1x1=False,
    ):
        """
        l_red_channels: list of reduction channels
        l_out_channels: list of output channels
        l_kernel_sizes: list of kernel sizes
        """
        super().__init__()
        self.branches = nn.ModuleList()
        for i, (f, c2, k) in enumerate(zip(fn, cn, kn)):
            if k == 1:
                layers = (
                    []
                    if f is None
                    else [
                        nn.MaxPool2d(3, 1, 1),
                    ]
                )
                layers.append(Conv1x1Layer(c1, c2, act=act))
                branch = nn.Sequential(*layers)
            else:
                branch = LightBottleneckLayer(
                    c1,
                    c2,
                    f,
                    k,
                    act=act,
                    norm=norm,
                )
            self.branches.append(branch)

    def forward(self, x):
        return torch.cat([branch(x) for branch in self.branches], dim=1)
