from einops.layers.torch import Reduce
from torch import einsum, nn

from .layers import (
    Conv1x1Layer,
    NormAct,
)


class AvgPool(nn.Module):
    def __init__(
        self,
        c1,
        c2=None,
        act="relu",
        norm="bn2d",
        drop=0.0,
        flatten=True,
    ):
        super().__init__()
        c2 = c2 or c1
        self.block = nn.Sequential(
            NormAct(c1, norm, act="noop"),
            nn.AdaptiveAvgPool2d(1),
            Conv1x1Layer(
                c1,
                c2,
                act=act,
                norm="noop",
            )
            if c2 != c1
            else nn.Identity(),
            nn.Flatten(1) if flatten else nn.Identity(),
            nn.Dropout(drop),
        )

    def forward(self, x):
        # [x: n, c, h, w]
        return self.block(x)  # [n, d, 1, 1] or [n, d]


class PatchNorm(nn.Module):
    def __init__(self, c1):
        super().__init__()

        self.block = nn.Sequential(
            nn.LayerNorm(c1),
            Reduce("n p c -> n c", "mean"),
        )

    def forward(self, x):
        return self.block(x)


class SequencePooling(nn.Module):
    def __init__(self, c1):
        super().__init__()
        self.attn_pool = nn.Sequential(
            nn.LayerNorm(c1),
            nn.Linear(c1, 1),
        )

    def forward(self, x):
        # x: [n, p, c]
        attn_weights = self.attn_pool(x).squeeze(-1)  # [n, p]
        return einsum(
            "n p, n p c -> n c",
            attn_weights.softmax(dim=1),
            x,
        )
