import math
from enum import Enum
from typing import Literal

import torch
import torch.nn as nn
from einops import rearrange
from object_detection_impl.models.blocks.core import (
    BasicFCLayer,
    Conv1x1Layer,
    ConvLayer,
    FlattenLayer,
)


class LearnablePositionEnc(nn.Module):
    def __init__(self, sizes):
        super().__init__()
        self.pos_enc = nn.Parameter(torch.zeros(1, *sizes))
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Parameter):
            nn.init.trunc_normal_(m, std=0.2)

    def forward(self, x):
        return x + self.pos_enc


class MultiHeadSA(nn.Module):
    def __init__(
        self,
        in_channels,
        num_heads=2,
        dropout=0.0,
        batch_first=True,
        norm="ln",
        norm_dims=None,
        **kwargs,
    ):
        super().__init__()
        self.stem = FlattenLayer(
            norm_dims=norm_dims or in_channels,
            norm=norm,
            norm_order="pre",
        )
        self.mha = nn.MultiheadAttention(
            in_channels,
            num_heads,
            dropout=dropout,
            batch_first=batch_first,
            **kwargs,
        )

    def forward(self, x):
        out = self.stem(x)
        out = self.mha(out, out, out)[0]  # [n, p, c]
        if len(x.shape) == 4:
            return rearrange(out, "n (h w) c -> n c h w", w=x.shape[-1])
        return out


class ECABlockV2(nn.Module):
    def __init__(self, in_channels, gamma=2, bias=1):
        super().__init__()
        k_size = int(abs(math.log(in_channels, 2) + bias) / gamma)
        k_size = k_size if k_size % 2 else k_size + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.block = nn.Conv1d(2, 1, k_size, padding="same")
        self.act = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.avg_pool(x).squeeze(-1).transpose(-1, -2)
        max_out = self.max_pool(x).squeeze(-1).transpose(-1, -2)
        weights = self.block(torch.cat([avg_out, max_out], dim=1))
        weights = self.act(weights).transpose(-1, -2).unsqueeze(-1)
        return x * weights


class ECABlock(nn.Module):
    def __init__(self, in_channels, gamma=2, bias=1):
        super().__init__()
        k_size = int(abs(math.log(in_channels, 2) + bias) / gamma)
        k_size = k_size if k_size % 2 else k_size + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.block = nn.Conv1d(1, 1, k_size, padding="same")
        self.act = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.avg_pool(x).squeeze(-1).transpose(-1, -2)
        weights = self.block(avg_out)
        weights = self.act(weights).transpose(-1, -2).unsqueeze(-1)
        return x * weights


class ChannelAttentionBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        reduction_rate=0.8,
        use_conv1x1=True,
    ):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        block_cls = Conv1x1Layer if use_conv1x1 else BasicFCLayer
        self.block = nn.Sequential(
            block_cls(
                in_channels,
                math.ceil(reduction_rate * in_channels),
                act="relu",
                norm="noop",
            ),
            block_cls(
                math.ceil(reduction_rate * in_channels),
                in_channels,
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
        self.block = ConvLayer(
            2,
            1,
            act="sigmoid",
            norm="noop",
        )

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return x * self.block(torch.cat([avg_out, max_out], dim=1))


class SEBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        reduction_rate: float = 0.8,
        use_conv1x1=True,
    ):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)

        block_cls = Conv1x1Layer if use_conv1x1 else BasicFCLayer
        self.block = nn.Sequential(
            block_cls(
                in_channels,
                math.ceil(reduction_rate * in_channels),
                act="relu",
                norm="noop",
            ),
            block_cls(
                math.ceil(reduction_rate * in_channels),
                in_channels,
                act="sigmoid",
                norm="noop",
            ),
        )

    def forward(self, x):
        weights = self.pool(x)
        weights = self.block(weights).view(x.size(0), -1, 1, 1)
        return x * weights


class CBAMBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        ca: Literal["eca", "ecav2", "cbam", "se"] = "eca",
        **ca_kwargs,
    ):
        super().__init__()
        if ca == "eca":
            self.ca = ECABlock(in_channels, **ca_kwargs)
        elif ca == "ecav2":
            self.ca = ECABlockV2(in_channels, **ca_kwargs)
        elif ca == "cbam":
            self.ca = ChannelAttentionBlock(in_channels, **ca_kwargs)
        elif ca == "se":
            self.ca = SEBlock(in_channels, **ca_kwargs)
        else:
            raise ValueError(f"unknown channel attn type: {ca}")

        self.sa = SpatialAttentionBlock()

    def forward(self, x):
        x = self.ca(x)
        return self.sa(x)


class Attentions(Enum):
    CA = ChannelAttentionBlock
    CBAM = CBAMBlock
    ECA = ECABlock
    ECAV2 = ECABlockV2
    SE = SEBlock
    ID = nn.Identity

    def __call__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        return self

    def create(self, **kwargs):
        local_kwargs = self.kwargs.copy()
        local_kwargs.update(kwargs)
        return self.value(*self.args, **local_kwargs)
