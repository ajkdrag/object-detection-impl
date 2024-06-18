import math

import numpy as np
import torch
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import nn


class Residual(nn.Module):
    def __init__(self, *layers, shortcut=None):
        super().__init__()
        self.shortcut = nn.Identity() if shortcut is None else shortcut
        self.residual = nn.Sequential(*layers)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return self.shortcut(x) + self.gamma * self.residual(x)


class SelfAttention(nn.Module):
    def __init__(self, dim, head_dim, heads=8, p_drop=0.0):
        super().__init__()
        inner_dim = head_dim * heads
        self.head_shape = (heads, head_dim)
        self.scale = head_dim**-0.5

        self.to_keys = nn.Linear(dim, inner_dim)
        self.to_queries = nn.Linear(dim, inner_dim)
        self.to_values = nn.Linear(dim, inner_dim)
        self.unifyheads = nn.Linear(inner_dim, dim)

        self.drop = nn.Dropout(p_drop)

    def forward(self, x):
        q_shape = x.shape[:-1] + self.head_shape

        keys = (
            self.to_keys(x).view(q_shape).transpose(1, 2)
        )  # move head forward to the batch dim
        queries = self.to_queries(x).view(q_shape).transpose(1, 2)
        values = self.to_values(x).view(q_shape).transpose(1, 2)

        att = queries @ keys.transpose(-2, -1)
        att = (att * self.scale).softmax(dim=-1)

        out = att @ values
        out = out.transpose(1, 2).contiguous().flatten(2)  # move head back
        out = self.unifyheads(out)
        out = self.drop(out)
        return out


class FeedForward(nn.Sequential):
    def __init__(self, dim, mlp_mult=4, p_drop=0.0):
        hidden_dim = dim * mlp_mult
        super().__init__(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(p_drop),
        )


class TransformerBlock(nn.Sequential):
    def __init__(self, dim, head_dim, heads, mlp_mult=4, p_drop=0.0):
        super().__init__(
            Residual(nn.LayerNorm(dim), SelfAttention(
                dim, head_dim, heads, p_drop)),
            Residual(nn.LayerNorm(dim), FeedForward(
                dim, mlp_mult, p_drop=p_drop)),
        )


class SoftSplit(nn.Module):
    def __init__(self, in_channels, dim, kernel_size=3, stride=2):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.unfold = nn.Unfold(kernel_size, stride=stride, padding=padding)
        self.project = nn.Linear(in_channels * kernel_size**2, dim)

    def forward(self, x):
        out = self.unfold(x).transpose(1, 2)
        out = self.project(out)
        return out


class Reshape(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        out = x.transpose(1, 2).unflatten(2, self.shape)
        return out


class T2TBlock(nn.Sequential):
    def __init__(
        self,
        image_size,
        token_dim,
        embed_dim,
        heads=1,
        mlp_mult=1,
        stride=2,
        p_drop=0.0,
    ):
        super().__init__(
            TransformerBlock(token_dim, token_dim // heads,
                             heads, mlp_mult, p_drop),
            Reshape((image_size, image_size)),
            SoftSplit(token_dim, embed_dim, stride=stride),
        )


class T2TModule(nn.Sequential):
    def __init__(
        self, in_channels, image_size, strides, token_dim, embed_dim, p_drop=0.0
    ):
        stride = strides[0]
        layers = [SoftSplit(in_channels, token_dim, stride=stride)]
        image_size = image_size // stride

        for stride in strides[1:-1]:
            layers.append(
                T2TBlock(image_size, token_dim, token_dim,
                         stride=stride, p_drop=p_drop)
            )
            image_size = image_size // stride

        stride = strides[-1]
        layers.append(
            T2TBlock(image_size, token_dim, embed_dim,
                     stride=stride, p_drop=p_drop)
        )

        super().__init__(*layers)


class TransformerBackbone(nn.Sequential):
    def __init__(self, dim, head_dim, heads, depth, mlp_mult=4, p_drop=0.0):
        layers = [
            TransformerBlock(dim, head_dim, heads, mlp_mult, p_drop)
            for _ in range(depth)
        ]
        super().__init__(*layers)


class Head(nn.Sequential):
    def __init__(self, dim, classes, p_drop=0.0):
        super().__init__(nn.LayerNorm(dim), nn.Dropout(p_drop), nn.Linear(dim, classes))


class TakeFirst(nn.Module):
    def forward(self, x):
        return x[:, 0]


class PositionEmbedding(nn.Module):
    def __init__(self, image_size, dim):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.zeros(1, image_size**2, dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x):
        # add positional embedding
        x = x + self.pos_embedding
        # add classification token
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        return x


class T2TViT(nn.Sequential):
    def __init__(
        self,
        cfg,
        strides=[2, 2, 2],
        token_dim=64,
        dim=128,
        head_dim=64,
        heads=4,
        backbone_depth=8,
        mlp_mult=2,
        in_channels=3,
        image_size=32,
        classes=10,
        trans_p_drop=0.2,
        head_p_drop=0.2,
    ):
        reduced_size = image_size // np.prod(strides)
        super().__init__(
            T2TModule(
                in_channels, image_size, strides, token_dim, dim, p_drop=trans_p_drop
            ),
            PositionEmbedding(reduced_size, dim),
            TransformerBackbone(
                dim,
                head_dim,
                heads,
                backbone_depth,
                mlp_mult=mlp_mult,
                p_drop=trans_p_drop,
            ),
            TakeFirst(),
            Head(dim, classes, p_drop=head_p_drop),
        )
