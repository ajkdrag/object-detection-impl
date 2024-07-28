import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange

from .layers import (
    BasicFCLayer,
    Conv1x1Layer,
    ConvLayer,
)


class MultiScaleSAV2(nn.Module):
    def __init__(
        self,
        c1,
        h,
        k_q=1,
        k_kv=1,
        s_q=1,
        s_kv=1,
        heads=2,
        drop=0.0,
        **kwargs,
    ):
        super().__init__()
        self.heads = heads
        self.h = h
        self.drop = drop
        self.pool_q = nn.Sequential(
            nn.MaxPool2d(k_q, s_q, (k_q - 1) //
                         2) if k_q > 1 else nn.Identity(),
            Rearrange("n c h w -> n (h w) c"),
        )
        self.pool_kv = nn.Sequential(
            nn.MaxPool2d(k_kv, s_kv, (k_kv - 1) //
                         2) if k_kv > 1 else nn.Identity(),
            Rearrange("n c h w -> n (h w) c"),
        )
        self.qkv = nn.Sequential(
            nn.Linear(
                c1,
                c1 * 3,
                bias=kwargs.get("qkv_bias", False),
            ),
            Rearrange(
                "n (h w) (a m d) -> a (n m) d h w",
                m=heads,
                a=3,
                h=h,
            ),
        )

    def forward(self, x):
        n, p, c = x.shape
        q, k, v = self.qkv(x)  # [n*m, c', h, w]
        q = self.pool_q(q)  # [n*m, (h'*w'), c']
        k = self.pool_kv(k)  # [n*m, (h''*w''), c']
        v = self.pool_kv(v)  # [n*m, (h''*w''), c']
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.drop,
        )  # [n*m, (h'*w'), c']
        return rearrange(out, "(n m) p d -> n p (m d)", m=self.num_heads)


class MultiHeadSA(nn.Module):
    def __init__(
        self,
        c1,
        heads=2,
        drop=0.0,
        **kwargs,
    ):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            c1,
            heads,
            dropout=drop,
            batch_first=True,
            **kwargs,
        )

    def forward(self, x):
        return self.mha(x, x, x)[0]  # [n, p, c]


class ECABlockV2(nn.Module):
    def __init__(self, c1, gamma=2, bias=1):
        super().__init__()
        k = int(abs(math.log(c1, 2) + bias) / gamma)
        k = k if k % 2 else k + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.block = nn.Conv1d(2, 1, k, padding="same")
        self.act = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.avg_pool(x).squeeze(-1).transpose(-1, -2)
        max_out = self.max_pool(x).squeeze(-1).transpose(-1, -2)
        weights = self.block(torch.cat([avg_out, max_out], dim=1))
        weights = self.act(weights).transpose(-1, -2).unsqueeze(-1)
        return x * weights


class ECABlock(nn.Module):
    def __init__(self, c1, gamma=2, bias=1):
        super().__init__()
        k = int(abs(math.log(c1, 2) + bias) / gamma)
        k = k if k % 2 else k + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.block = nn.Conv1d(1, 1, k, padding="same")
        self.act = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.avg_pool(x).squeeze(-1).transpose(-1, -2)
        weights = self.block(avg_out)
        weights = self.act(weights).transpose(-1, -2).unsqueeze(-1)
        return x * weights


class ChannelAttentionBlock(nn.Module):
    def __init__(self, c1, f=0.8, use_conv=True):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        block_cls = Conv1x1Layer if use_conv else BasicFCLayer
        self.block = nn.Sequential(
            block_cls(
                c1,
                math.ceil(f * c1),
                act="relu",
                norm="noop",
            ),
            block_cls(
                math.ceil(f * c1),
                c1,
                act="noop",
                norm="noop",
            ),
        )
        self.act = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.block(self.avg_pool(x))
        max_out = self.block(self.max_pool(x))
        return x * self.act(avg_out + max_out)


class SpatialAttentionBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = ConvLayer(2, 1, act="sigmoid", norm="noop")

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return x * self.block(torch.cat([avg_out, max_out], dim=1))


class SEBlock(nn.Module):
    def __init__(self, c1, f=0.25, use_conv=True):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)

        block_cls = Conv1x1Layer if use_conv else BasicFCLayer
        self.block = nn.Sequential(
            block_cls(c1, math.ceil(f * c1), act="relu", norm="noop"),
            block_cls(math.ceil(f * c1), c1, act="hsigmoid", norm="noop"),
        )

    def forward(self, x):
        weights = self.pool(x)
        weights = self.block(weights).view(x.size(0), -1, 1, 1)
        return x * weights


class CBAMBlock(nn.Module):
    def __init__(
        self,
        c1,
        ca: Literal["eca", "ecav2", "cbam", "se"] = "eca",
        **ca_kwargs,
    ):
        super().__init__()
        if ca == "eca":
            self.ca = ECABlock(c1, **ca_kwargs)
        elif ca == "ecav2":
            self.ca = ECABlockV2(c1, **ca_kwargs)
        elif ca == "cbam":
            self.ca = ChannelAttentionBlock(c1, **ca_kwargs)
        elif ca == "se":
            self.ca = SEBlock(c1, **ca_kwargs)
        else:
            raise ValueError(f"unknown channel attn type: {ca}")

        self.sa = SpatialAttentionBlock()

    def forward(self, x):
        x = self.ca(x)
        return self.sa(x)
