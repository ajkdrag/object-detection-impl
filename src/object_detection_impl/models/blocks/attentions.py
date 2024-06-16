import math
from enum import Enum
from typing import Literal

import torch
import torch.nn as nn
from object_detection_impl.models.blocks.activations import Activations
from object_detection_impl.models.blocks.core import (
    BasicFCLayer,
    Conv1x1Layer,
    ConvLayer,
    ExpansionFCLayer,
    FlattenLayer,
    ShortcutLayer,
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


class MultiHead_SA(nn.Module):
    def __init__(
        self,
        in_channels,
        num_heads=2,
        dropout=0.0,
        expansion=4,
        activation=Activations.GELU(),
    ):
        super().__init__()
        self.norm = nn.LayerNorm(in_channels)
        self.stem = FlattenLayer()
        self.mha = nn.MultiheadAttention(
            in_channels,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.scale = nn.Parameter(torch.zeros(1))
        self.shortcut = ShortcutLayer(in_channels, in_channels)
        if expansion is not None:
            self.mlp_flag = 1
            self.mlp = nn.Sequential(
                self.norm,
                ExpansionFCLayer(
                    in_channels,
                    expansion=expansion,
                    activation=activation,
                    dropout=dropout,
                    bn=False,
                ),
            )
        else:
            self.mlp_flag = 0
            self.mlp = activation.create()

    def forward(self, x):
        normed = self.norm(self.stem(x))  # [n, p, c]
        attn_out, _ = self.mha(normed, normed, normed)
        mha_out = self.scale * attn_out + self.stem(self.shortcut(x))
        out = self.mlp(mha_out) + self.mlp_flag * (mha_out)

        if len(x.shape) == 4:
            return out.transpose(1, 2).reshape(x.shape)
        return out


class ECA_V2_Block(nn.Module):
    def __init__(self, in_channels, gamma=2, bias=1):
        super().__init__()
        k_size = int(abs(math.log(in_channels, 2) + bias) / gamma)
        k_size = k_size if k_size % 2 else k_size + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.block = nn.Conv1d(2, 1, k_size, padding="same")
        self.activation = Activations.SIGMOID().create()

    def forward(self, x):
        avg_out = self.avg_pool(x).squeeze(-1).transpose(-1, -2)
        max_out = self.max_pool(x).squeeze(-1).transpose(-1, -2)
        weights = self.block(torch.cat([avg_out, max_out], dim=1))
        weights = self.activation(weights).transpose(-1, -2).unsqueeze(-1)
        return x * weights


class ECA_Block(nn.Module):
    def __init__(self, in_channels, gamma=2, bias=1):
        super().__init__()
        k_size = int(abs(math.log(in_channels, 2) + bias) / gamma)
        k_size = k_size if k_size % 2 else k_size + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.block = nn.Conv1d(1, 1, k_size, padding="same")
        self.activation = Activations.SIGMOID().create()

    def forward(self, x):
        avg_out = self.avg_pool(x).squeeze(-1).transpose(-1, -2)
        weights = self.block(avg_out)
        weights = self.activation(weights).transpose(-1, -2).unsqueeze(-1)
        return x * weights


class CBAM_CA_Block(nn.Module):
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
        squeeze_kwargs = (
            {"bn": False}
            if use_conv1x1
            else {
                "bn": False,
                "dropout": 0.0,
            }
        )
        excitation_kwargs = (
            {"bn": False}
            if use_conv1x1
            else {
                "bn": False,
                "dropout": 0.0,
            }
        )
        self.block = nn.Sequential(
            block_cls(
                in_channels,
                math.ceil(reduction_rate * in_channels),
                activation=Activations.RELU(inplace=True),
                **squeeze_kwargs,
            ),
            block_cls(
                math.ceil(reduction_rate * in_channels),
                in_channels,
                activation=Activations.ID(),
                **excitation_kwargs,
            ),
        )
        self.activation = Activations.SIGMOID().create()

    def forward(self, x):
        avg_out = self.block(self.avg_pool(x))
        max_out = self.block(self.max_pool(x))
        return x * self.activation(avg_out + max_out)


class CBAM_SA_Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = ConvLayer(
            2,
            1,
            activation=Activations.SIGMOID(),
            bn=False,
        )

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return x * self.block(torch.cat([avg_out, max_out], dim=1))


class CBAM_Block(nn.Module):
    def __init__(
        self,
        in_channels,
        ca: Literal["eca", "ecav2", "cbam", "se"] = "eca",
        **ca_kwargs,
    ):
        super().__init__()
        if ca == "eca":
            self.ca = ECA_Block(in_channels, **ca_kwargs)
        elif ca == "ecav2":
            self.ca = ECA_V2_Block(in_channels, **ca_kwargs)
        elif ca == "cbam":
            self.ca = CBAM_CA_Block(in_channels, **ca_kwargs)
        elif ca == "se":
            self.ca = SEBlock(in_channels, **ca_kwargs)
        else:
            raise ValueError(f"unknown channel attn type: {ca}")

        self.sa = CBAM_SA_Block()

    def forward(self, x):
        x = self.ca(x)
        return self.sa(x)


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
        squeeze_kwargs = (
            {"bn": False}
            if use_conv1x1
            else {
                "bn": False,
                "dropout": 0.0,
            }
        )
        excitation_kwargs = (
            {"bn": False}
            if use_conv1x1
            else {
                "bn": False,
                "dropout": 0.0,
            }
        )
        self.block = nn.Sequential(
            block_cls(
                in_channels,
                math.ceil(reduction_rate * in_channels),
                activation=Activations.RELU(inplace=True),
                **squeeze_kwargs,
            ),
            block_cls(
                math.ceil(reduction_rate * in_channels),
                in_channels,
                activation=Activations.SIGMOID(),
                **excitation_kwargs,
            ),
        )

    def forward(self, x):
        weights = self.pool(x)
        weights = self.block(weights).view(x.size(0), -1, 1, 1)
        return x * weights


class Attentions(Enum):
    CBAM_CA = CBAM_CA_Block
    CBAM = CBAM_Block
    ECA = ECA_Block
    ECAV2 = ECA_V2_Block
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
