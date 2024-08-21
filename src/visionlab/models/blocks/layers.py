import math

import torch
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from object_detection_impl.models.act_factory import Acts
from object_detection_impl.models.norm_factory import Norms
from torch import nn


class Shifting(nn.Module):
    def __init__(self, shift):
        super().__init__()
        self.shift = shift

    def forward(self, x):
        x_pad = F.pad(x, (self.shift, self.shift, self.shift, self.shift))
        x_lu = x_pad[:, :, : -self.shift * 2, : -self.shift * 2]
        x_ru = x_pad[:, :, : -self.shift * 2, self.shift * 2 :]
        x_lb = x_pad[:, :, self.shift * 2 :, : -self.shift * 2]
        x_rb = x_pad[:, :, self.shift * 2 :, self.shift * 2 :]
        x_cat = torch.cat([x, x_lu, x_ru, x_lb, x_rb], dim=1)
        return x_cat


class NormAct(nn.Module):
    def __init__(self, c1, norm="bn2d", act="relu"):
        super().__init__()
        self.block = nn.Sequential(
            Norms.get(norm, c1),
            Acts.get(act),
        )

    def forward(self, x):
        return self.block(x)


class ApplyNorm(nn.Module):
    def __init__(self, norm, fn, pre=True):
        super().__init__()
        layers = [norm, fn] if pre else [fn, norm]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class BasicFCLayer(nn.Module):
    def __init__(
        self,
        c1,
        c2=None,
        act="relu",
        norm="ln",
        drop=0.0,
        pre_norm=True,
    ):
        super().__init__()
        c2 = c2 or c1
        fc_layers = [
            nn.Linear(c1, c2),
            Acts.get(act),
            nn.Dropout(drop),
        ]
        self.norm = norm

        norm_dims = c1 if pre_norm else c2
        self.block = ApplyNorm(
            Norms.get(norm, norm_dims),
            nn.Sequential(*fc_layers),
            pre_norm,
        )

    def forward(self, x):
        return self.block(x)


class ExpansionFCLayer(nn.Module):
    def __init__(
        self,
        c1,
        c2=None,
        f=4,
        act="relu",
        norm="ln",
        drop=0.0,
        pre_norm=True,
    ):
        super().__init__()
        c2 = c2 or c1
        self.block = ApplyNorm(
            Norms.get(norm, c1 if pre_norm else c2),
            nn.Sequential(
                BasicFCLayer(
                    c1,
                    math.ceil(c1 * f),
                    act=act,
                    drop=0.0,
                    norm="noop",
                ),
                BasicFCLayer(
                    math.ceil(c1 * f),
                    c2,
                    act="noop",
                    drop=drop,
                    norm="noop",
                ),
            ),
            pre_norm,
        )

    def forward(self, x):
        return self.block(x)


class SoftSplit(nn.Module):
    def __init__(self, c1, c2=None, k=3, s=1, proj=True):
        super().__init__()
        c2 = c2 or c1
        self.block = nn.Sequential(
            nn.Unfold(kernel_size=k, stride=s, padding=(k - 1) // 2),
            Rearrange("n c p -> n p c"),
            nn.Linear(c1 * k**2, c2) if proj else nn.Identity(),
        )

    def forward(self, x):
        # ip: [n, c, w, h], op = [n, p, c]
        # p = (h - k + 2 * padding) / s + 1)**2
        return self.block(x)


class FlattenLayer(nn.Module):
    def __init__(self, c1=None, norm="noop", pre_norm=True):
        super().__init__()
        self.block = ApplyNorm(
            Norms.get(norm, c1),
            nn.Identity(),
            pre_norm,
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
        c1,
        c2=None,
        k=3,
        s=1,
        p="same",
        g=1,
        act="relu",
        norm="bn2d",
        pre_normact=False,
        **kwargs,
    ):
        super().__init__()
        c2 = c2 or c1
        if s > 1 and p == "same":
            p = (k - 1) // 2
        layers = [
            nn.Conv2d(
                c1,
                c2,
                kernel_size=k,
                padding=p,
                stride=s,
                groups=g,
                **kwargs,
            ),
            NormAct(c1 if pre_normact else c2, norm, act),
        ]
        if pre_normact:
            layers.reverse()
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class DWConvLayer(ConvLayer):
    def __init__(self, c1, c2=None, k=3, s=1, **kwargs):
        assert c2 % c1 == 0, f"c2 {c2} must be divisible by c1 {c1}"
        super().__init__(c1, c2, k=k, s=s, g=c1, **kwargs)


class DWSepConvLayer(nn.Sequential):
    def __init__(self, c1, c2=None, k=3, s=1, pw_act="noop", **kwargs):
        c2 = c2 or c1
        super().__init__(
            DWConvLayer(c1, c1, k, s, **kwargs),
            Conv1x1Layer(c1, c2, act=pw_act),
        )


class Conv1x1Layer(ConvLayer):
    def __init__(self, c1, c2=None, s=1, **kwargs):
        super().__init__(c1, c2, k=1, s=s, **kwargs)


class DownsampleConvLayer(ConvLayer):
    def __init__(self, c1, c2=None, k=3, s=2, **kwargs):
        super().__init__(c1, c2, k=k, s=s, **kwargs)


class DownsamplePoolLayer(nn.Module):
    def __init__(self, c1, c2=None, pool="avg", **kwargs):
        super().__init__()
        self.conv = Conv1x1Layer(c1, c2, **kwargs)
        if pool == "avg":
            self.pool = nn.AvgPool2d(k=2, s=2)
        elif pool == "max":
            self.pool = nn.MaxPool2d(k=2, s=2)
        else:
            raise ValueError(f"Unsupported pool type: {pool}")

    def forward(self, x):
        return self.pool(self.conv(x))


class ShortcutLayer(nn.Module):
    def __init__(self, c1, c2=None, s=1):
        super().__init__()
        c2 = c2 or c1
        if c1 == c2 and s == 1:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = Conv1x1Layer(c1, c2, s=s, act="noop")

    def forward(self, x):
        return self.shortcut(x)


class LightShortcutLayer(nn.Module):
    def __init__(self, c1, c2=None, k=3, s=1, g=1):
        super().__init__()
        c2 = c2 or c1
        if c1 == c2 and s == 1:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.AvgPool2d(k, stride=s, padding=1),
                Conv1x1Layer(c1, c2, g=g, act="noop"),
            )

    def forward(self, x):
        return self.shortcut(x)


class Shortcut3x3Layer(nn.Module):
    def __init__(self, c1, c2=None, s=1):
        super().__init__()
        c2 = c2 or c1
        if c1 == c2 and s == 1:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = ConvLayer(c1, c2, s=s, act="noop")

    def forward(self, x):
        return self.shortcut(x)


class GhostLayer(nn.Module):
    def __init__(
        self,
        c1,
        c2=None,
        k=3,
        s=1,
        act="relu",
        norm="bn2d",
    ):
        super().__init__()
        c2 = c2 or c1
        assert c2 % 2 == 0, "Output channels should be even"
        c_ = c2 // 2
        self.block_1 = Conv1x1Layer(c1, c_, act=act, norm=norm)
        self.block_2 = DWConvLayer(c_, c_, k=k, s=s, act=act, norm=norm)

    def forward(self, x):
        x1 = self.block_1(x)
        return torch.cat([x1, self.block_2(x1)], dim=1)


class DWSepShortcutLayer(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, act="relu", norm="bn2d"):
        super().__init__()
        if c1 == c2 and s == 1:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = DWSepConvLayer(
                c1,
                c2,
                k,
                s,
                act=act,
                pw_act="noop",
                norm=norm,
            )

    def forward(self, x):
        return self.shortcut(x)


class BottleneckLayer(nn.Sequential):
    def __init__(
        self,
        c1,
        c2=None,
        k=3,
        s=1,
        act="relu",
        norm="bn2d",
    ):
        c2 = c2 or c1
        super().__init__(
            ConvLayer(c1, c2, k=k, s=s, act=act, norm=norm),
            ConvLayer(c2, c2, act="noop", norm=norm),
        )


class LightBottleneckLayer(nn.Sequential):
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
        pw_act="noop",
        first_1x1_optional=True,
    ):
        c2 = c2 or c1
        c_ = math.ceil(f * c1)
        super().__init__(
            Conv1x1Layer(
                c1,
                c_,
                act=act,
                norm=norm,
            )
            if f != 1 and first_1x1_optional
            else nn.Identity(),
            ConvLayer(c_, c_, k, s, g=g, act=act, norm=norm),
            Conv1x1Layer(c_, c2, act=pw_act, norm=norm),
        )


class DWLightBottleneckLayer(LightBottleneckLayer):
    def __init__(
        self,
        c1,
        c2=None,
        f=4,
        k=3,
        s=1,
        act="relu",
        norm="bn2d",
        **kwargs,
    ):
        g = math.ceil(f * c1)
        super().__init__(c1, c2, f, k, s, g, act, norm, **kwargs)


class DenseShortcutLayer(nn.Module):
    def __init__(
        self,
        c1,
        f=32,
        act="relu",
        norm="bn2d",
    ):
        super().__init__()
        self.block = ConvLayer(c1, f, act=act, norm=norm, pre_normact=True)

    def forward(self, x):
        out = self.block(x)
        return torch.cat([x, out], dim=1)


class ScaledResidual(nn.Module):
    def __init__(self, *layers, shortcut=None, apply_shortcut=True):
        super().__init__()
        self.shortcut = nn.Identity() if shortcut is None else shortcut
        self.residual = nn.Sequential(*layers)
        self.apply_shortcut = int(apply_shortcut)
        self.gamma = nn.Parameter(torch.zeros(1)) if apply_shortcut else 1

    def forward(self, x):
        residual = self.gamma * self.residual(x)
        if self.apply_shortcut:
            return self.shortcut(x) + residual
        return residual


class ChannelShuffle(nn.Module):
    def __init__(self, g):
        super().__init__()
        self.g = g
        self.block = Rearrange("n (g d) h w -> n (d g) h w", g=g)

    def forward(self, x):
        return self.block(x)
