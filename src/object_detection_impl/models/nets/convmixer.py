import torch.nn as nn
from einops.layers.torch import Rearrange
from object_detection_impl.models.blocks.composites import (
    ConvMixerBlock,
)
from object_detection_impl.utils.registry import load_obj
from omegaconf import DictConfig


class ConvMixerTiny(nn.Module):
    def __init__(self, config: DictConfig) -> None:
        super().__init__()
        n_layers = 8
        embed_sz = 256
        c = 3

        self.stem = load_obj(config.model.stem.class_name)(
            in_channels=c,
            out_channels=embed_sz,
            **config.model.stem.get("params", {}),
        )

        self.trunk = nn.Sequential(
            *[
                ConvMixerBlock(
                    embed_sz,
                    kernel_size=5,
                )
                for _ in range(n_layers)
            ],
            Rearrange("n c h w -> n (h w) c"),
        )

        self.neck = load_obj(config.model.neck.class_name)(
            in_channels=embed_sz,
            **config.model.neck.get("params", {}),
        )

        self.head = load_obj(config.model.head.class_name)(
            in_features=embed_sz,
            **config.model.head.params,
        )

    def forward(self, x):
        return self.head(self.neck(self.trunk(self.stem(x))))
