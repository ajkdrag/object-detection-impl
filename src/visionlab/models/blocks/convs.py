import math

import torch
import torch.nn as nn

from ...utils.ml import make_divisible
from ..act_factory import Acts
from ..attn_factory import Attentions
from ..norm_factory import Norms
from .layers import (
    BottleneckLayer,
    ChannelShuffle,
    Conv1x1Layer,
    ConvLayer,
    DenseShortcutLayer,
    DWConvLayer,
    DWLightBottleneckLayer,
    DWSepConvLayer,
    DWSepShortcutLayer,
    GhostLayer,
    LightBottleneckLayer,
    LightShortcutLayer,
    ScaledResidual,
    Shortcut3x3Layer,
    ShortcutLayer,
)


class ConvNextBlock(nn.Module):
    def __init__(self, c1, c2=None, f=2, k=3, s=1):
        super().__init__()
        c2 = c2 or c1
        c_ = make_divisible(f * c1, 4)
        self.block = ScaledResidual(
            DWConvLayer(c1, c1, k, s, act="noop", norm="noop"),
            Norms.get("lnch", c1),
            Conv1x1Layer(c1, c_, act="gelu", norm="noop"),
            Norms.get("grn", c_),
            Conv1x1Layer(c_, c2, act="noop", norm="noop"),
            shortcut=ShortcutLayer(c1, c2, 1),
        )

    def forward(self, x):
        return self.block(x)


class ShuffleNetV2Block(nn.Module):
    def __init__(self, c1, c2=None, k=3, s=1, act="relu"):
        super().__init__()
        c2 = c2 or c1
        self.s = s
        c_ = c2 // 2

        if s == 1:
            assert (
                c1 == c2 and c1 % 2 == 0
            ), "op channels should be equal to ip channels and even, for stride == 1"
            c1 = c1 // 2
            self.branch_1 = nn.Identity()
        else:
            self.branch_1 = DWSepConvLayer(
                c1,
                c_,
                k=k,
                s=s,
                pw_act=act,
                act="noop",
            )

        self.branch_2 = DWLightBottleneckLayer(
            c1,
            c_,
            f=1,
            k=k,
            s=s,
            pw_act=act,
            act="noop",
            first_1x1_optional=False,
        )
        self.shuffle = ChannelShuffle(g=2)

    def forward(self, x):
        x1, x2 = x, x
        if self.s == 1:
            x1, x2 = x.chunk(2, dim=1)
        out = torch.cat((self.branch_1(x1), self.branch_2(x2)), dim=1)
        return self.shuffle(out)


class ShuffleNetBlock(nn.Module):
    def __init__(self, c1, c2=None, f=2, k=3, s=1, g=3, act="relu"):
        super().__init__()
        self.s = s
        c2 = c2 or c1
        c2 = c2 // 2 if s != 1 else c2
        c_ = make_divisible(f * c1, 4)

        self.block = nn.Sequential(
            Conv1x1Layer(c1, c_, g=g, act=act),
            ChannelShuffle(g),
            ConvLayer(c_, c_, k=k, s=s, g=g, act="noop"),
            Conv1x1Layer(c_, c2, g=g, act="noop"),
        )
        self.shortcut = LightShortcutLayer(c1, c2, s=s, g=g)

    def forward(self, x):
        out = self.block(x)
        shortcut = self.shortcut(x)

        if self.s != 1:
            return torch.cat([shortcut, out], 1)
        return shortcut + out


class FusedMBConvBlock(nn.Module):
    def __init__(self, c1, c2, f=4, k=3, s=1, act="silu"):
        super().__init__()

        c_ = math.ceil(f * c1)

        self.block = ScaledResidual(
            ConvLayer(c1, c_, k=k, s=s, act=act) if f != 1 else nn.Identity(),
            Conv1x1Layer(c_, c2, act="noop") if f != 1 else nn.Identity(),
            shortcut=ShortcutLayer(c1, c2, s),
            apply_shortcut=(s == 1 and c1 == c2),
        )

    def forward(self, x):
        return self.block(x)


class MBConvBlock(nn.Module):
    def __init__(
        self,
        c1,
        c2,
        f=4,
        k=3,
        s=1,
        act="hswish",
        attn="se",
    ):
        super().__init__()
        c_ = math.ceil(f * c1)

        self.block = ScaledResidual(
            Conv1x1Layer(c1, c_, act=act) if f != 1 else nn.Identity(),
            DWConvLayer(c_, c_, k=k, s=s, act=act),
            Attentions.get(attn, c_),
            Conv1x1Layer(c_, c2, act="noop"),
            shortcut=ShortcutLayer(c1, c2, s),
            apply_shortcut=(s == 1 and c1 == c2),
        )

    def forward(self, x):
        return self.block(x)


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
                apply_shortcut=(s == 1 and c1 == c2),
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
