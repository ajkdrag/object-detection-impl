import torch.nn as nn
from einops.layers.torch import Rearrange

from .layers import (
    BasicFCLayer,
    ConvLayer,
    SoftSplit,
    UnflattenLayer,
)
from .transformers import (
    T2TBlock,
)


class T2TPatchV2(nn.Module):
    def __init__(
        self,
        c1,
        c2,
        h,
        f=1,
        kn=3,
        sn=1,
        heads=1,
        act="gelu",
        drop=0.0,
        **kwargs,
    ):
        super().__init__()
        layers = []
        last = len(sn) - 1
        c_ = int(f * c2)

        for i, (k, s) in enumerate(zip(kn, sn)):
            layers.extend(
                [
                    SoftSplit(c1, c2 if i == last else c_, k, s)
                    if i == 0
                    else T2TBlock(
                        c_,
                        c2 if i == last else c_,
                        h,
                        f=f,
                        k=k,
                        s=s,
                        heads=heads,
                        act=act,
                        drop=drop,
                    ),
                ]
            )
            h //= s
        layers.append(UnflattenLayer(h=h))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class MDMLPPatch(nn.Module):
    def __init__(
        self,
        c1,
        c2,
        k,
        s,
        act="noop",
        norm="noop",
        drop=0.0,
        **kwargs,
    ):
        super().__init__()
        self.block = nn.Sequential(
            nn.Unfold(kernel_size=k, stride=s, padding=(k - 1) // 2),
            Rearrange(
                "n (c d) p -> n c p d",
                c=c1,
                d=k**2,
            ),
            BasicFCLayer(
                k**2,
                c2,
                act=act,
                norm=norm,
                drop=drop,
                pre_norm=False,
            ),
        )

    def forward(self, x):
        return self.block(x)


class ConvMpPatch(nn.Module):
    def __init__(
        self,
        c1,
        c2,
        k=3,
        s=1,
        act="noop",
        norm="noop",
        mp=True,
        **kwargs,
    ):
        super().__init__()
        self.block = nn.Sequential(
            ConvLayer(c1, c2, k, s, act=act, norm=norm),
            nn.MaxPool2d(
                kernel_size=kwargs.get("mp_k", 3),
                stride=kwargs.get("mp_s", 2),
                padding=kwargs.get("mp_p", 1),
            )
            if mp
            else nn.Identity(),
        )

    def forward(self, x):
        return self.block(x)
